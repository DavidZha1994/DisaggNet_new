import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any


# 极简推荐配置占位（保持导入兼容，但不再使用复杂分支）
RECOMMENDED_LOSS_CONFIGS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "regression_weight": 1.0,
        "classification_weight": 1.0,
        "conservation_weight": 0.0,
        "unknown_weight": 0.0,
        "unknown_match_weight": 1.0,
        "unknown_l1_penalty": 0.1,
        "huber_delta": 1.0,
        "normalize_per_device": True,
        "active_threshold_rel": 0.05,
        "off_penalty_weight": 0.25,
    }
}


class NILMLoss:
    """
    简洁的 NILM 联合损失：
    - 序列回归（相对刻度的 SmoothL1 + 仅在激活时刻的相对误差项 + 关闭时抑制项）
    - 序列分类（逐时间步 BCE，支持 per-device pos_weight）
    - 能量守恒（窗口级，相对误差）
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.regression_weight = float(cfg.get("regression_weight", 1.0))
        self.classification_weight = float(cfg.get("classification_weight", 1.0))
        self.conservation_weight = float(cfg.get("conservation_weight", 0.0))
        self.unknown_weight = float(cfg.get("unknown_weight", 0.0))
        self.unknown_match_weight = float(cfg.get("unknown_match_weight", 1.0))
        self.unknown_l1_penalty = float(cfg.get("unknown_l1_penalty", 0.1))
        self.huber_delta = float(cfg.get("huber_delta", 1.0))
        self.normalize_per_device = bool(cfg.get("normalize_per_device", True))
        # 更低的激活阈值，提升对中低功率事件的敏感度
        self.active_threshold_rel = float(cfg.get("active_threshold_rel", 0.02))
        self.off_penalty_weight = float(cfg.get("off_penalty_weight", 0.25))
        # 相对误差权重与激活增强权重，提高模型对事件期的学习强度
        self.rel_loss_weight = float(cfg.get("rel_loss_weight", 1.5))
        self.active_boost_weight = float(cfg.get("active_boost_weight", 2.0))
        # 稀有设备活跃期加权：根据训练集中该设备的阳性比例 p_k 进行增强
        # device_weight = (1 / (p_k + eps)) ** rare_active_alpha，乘以 rare_active_floor 作为下限系数
        self.rare_active_alpha = float(cfg.get("rare_active_alpha", 0.0))
        self.rare_active_floor = float(cfg.get("rare_active_floor", 1.0))
        # 稀有负样本增强（针对近似“常开”的设备，如冰箱）：
        # 当设备的阳性比例 p_k 较高时，负样本（关闭期）很稀有，容易被忽略。
        # 使用 device_neg_weight = (1 / (1 - p_k + eps)) ** rare_neg_alpha * rare_neg_floor
        # 放大 y=0 项的损失，促使模型学习到“关机期”而不至于全为 1。
        self.rare_neg_alpha = float(cfg.get("rare_neg_alpha", 0.8))
        self.rare_neg_floor = float(cfg.get("rare_neg_floor", 1.0))
        # 分类损失类型与参数（默认 BCE，可选 focal）
        self.classification_loss_type = str(cfg.get("classification_loss_type", "bce")).lower()
        self.focal_alpha = float(cfg.get("focal_alpha", 0.25))
        self.focal_gamma = float(cfg.get("focal_gamma", 2.0))
        # 二值化训练（Straight-Through Estimator）
        self.classification_hard = bool(cfg.get("classification_hard", False))
        self.hard_threshold = float(cfg.get("hard_threshold", 0.5))
        # 关闭期抑制仅在目标功率趋近于零时施加（相对刻度阈值）
        self.off_penalty_rel_threshold = float(cfg.get("off_penalty_rel_threshold", 0.02))
        # 可选：分类 pos_weight（每设备）
        self.pos_weight_vec: Optional[torch.Tensor] = None
        # 可选：每设备阳性比例 p（用于稀有设备加权）
        self.prior_p_vec: Optional[torch.Tensor] = None

    def set_pos_weight(self, pos_weight: Optional[torch.Tensor]) -> None:
        self.pos_weight_vec = None if pos_weight is None else pos_weight.detach().clone().float()

    def set_prior_p(self, prior_p: Optional[torch.Tensor]) -> None:
        self.prior_p_vec = None if prior_p is None else prior_p.detach().clone().float()

    def _ensure_scale(self, power_scale: Optional[torch.Tensor], device: torch.device, K: int) -> torch.Tensor:
        if not isinstance(power_scale, torch.Tensor):
            return torch.ones(1, 1, K, dtype=torch.float32, device=device)
        scale = torch.clamp(power_scale.to(device), min=1e-6).view(1, 1, -1)
        return scale

    def regression_seq_loss(self, pred_seq: torch.Tensor, target_seq: torch.Tensor,
                             status_seq: Optional[torch.Tensor], valid_mask: Optional[torch.Tensor],
                             power_scale: Optional[torch.Tensor]) -> torch.Tensor:
        """
        在相对刻度上计算 SmoothL1 + 激活时相对误差；并在关闭时添加抑制项。
        - pred_seq/target_seq: (B, L, K)
        - status_seq: (B, L, K) 或 None（为 None 则不使用激活门控与抑制项）
        - valid_mask: (B, L, K) 有效位置布尔掩码
        - power_scale: (K,) 每设备尺度（例如P95）
        """
        B, L, K = pred_seq.size(0), pred_seq.size(1), pred_seq.size(2)
        device = pred_seq.device
        scale = self._ensure_scale(power_scale, device, K)
        pred_n = pred_seq / scale
        target_n = target_seq / scale

        # 有效掩码
        valid = valid_mask if isinstance(valid_mask, torch.Tensor) else torch.isfinite(pred_n) & torch.isfinite(target_n)
        valid = valid.to(torch.bool)

        # 计算稀有设备权重（形状 1x1xK），仅当提供 prior_p 且 alpha>0 时启用
        device_weight = torch.ones(1, 1, K, dtype=torch.float32, device=device)
        if isinstance(self.prior_p_vec, torch.Tensor) and self.prior_p_vec.numel() == K and (self.rare_active_alpha > 0.0):
            p = torch.clamp(self.prior_p_vec.to(device), min=1e-6).view(1, 1, -1)
            device_weight = torch.pow(1.0 / p, self.rare_active_alpha) * float(self.rare_active_floor)

        # SmoothL1（Huber）在相对刻度
        delta = self.huber_delta
        resid = torch.abs(pred_n - target_n)
        huber_el = torch.where(resid < delta, 0.5 * resid ** 2, delta * (resid - 0.5 * delta))
        # 激活增强：在设备开关状态为1时，放大损失贡献，避免“全零预测但损失很小”的现象
        if isinstance(status_seq, torch.Tensor):
            boost = 1.0 + self.active_boost_weight * device_weight * torch.clamp(status_seq, min=0.0, max=1.0)
            huber_el = huber_el * boost
        huber_el = torch.where(valid, huber_el, torch.zeros_like(huber_el))
        denom = valid.float().sum().clamp_min(1.0)
        huber_loss = huber_el.sum() / denom

        # 激活门控下的相对误差（仅在 target_n 大于阈值时计算）
        rel_thr = self.active_threshold_rel
        act_mask = (target_n > rel_thr) & valid
        rel_err = torch.abs(pred_n - target_n) / (target_n.abs() + 1e-6)
        if isinstance(status_seq, torch.Tensor):
            boost = 1.0 + self.active_boost_weight * device_weight * torch.clamp(status_seq, min=0.0, max=1.0)
            rel_err = rel_err * boost
        rel_err = torch.where(act_mask, rel_err, torch.zeros_like(rel_err))
        denom_rel = act_mask.float().sum().clamp_min(1.0)
        rel_loss = rel_err.sum() / denom_rel

        # 关闭抑制项：当 status==0 且目标功率接近零时惩罚过高的预测（避免误抑制基线）
        off_pen = torch.tensor(0.0, device=device)
        if isinstance(status_seq, torch.Tensor):
            off_mask = (status_seq <= 0.5) & valid & (target_n <= float(self.off_penalty_rel_threshold))
            # 在原始刻度下惩罚预测值（L1，使用非负幅值避免负损失）
            off_mag = torch.where(off_mask, torch.relu(pred_seq), torch.zeros_like(pred_seq))
            denom_off = off_mask.float().sum().clamp_min(1.0)
            off_pen = off_mag.sum() / denom_off

        return huber_loss + self.rel_loss_weight * rel_loss + self.off_penalty_weight * off_pen

    def regression_seq_loss_per_device(self, pred_seq: torch.Tensor, target_seq: torch.Tensor,
                                       status_seq: Optional[torch.Tensor], valid_mask: Optional[torch.Tensor],
                                       power_scale: Optional[torch.Tensor]) -> torch.Tensor:
        """按设备返回序列回归损失 (K,)。与 regression_seq_loss 相同的组成，但对每个设备独立归约。
        - 返回值：shape (K,) ，每列为该设备在整个批次与时间维上的平均损失。
        """
        B, L, K = pred_seq.size(0), pred_seq.size(1), pred_seq.size(2)
        device = pred_seq.device
        scale = self._ensure_scale(power_scale, device, K)
        pred_n = pred_seq / scale
        target_n = target_seq / scale

        valid = valid_mask if isinstance(valid_mask, torch.Tensor) else (torch.isfinite(pred_n) & torch.isfinite(target_n))
        valid = valid.to(torch.bool)

        # 稀有设备权重（形状 1x1xK）
        device_weight = torch.ones(1, 1, K, dtype=torch.float32, device=device)
        if isinstance(self.prior_p_vec, torch.Tensor) and self.prior_p_vec.numel() == K and (self.rare_active_alpha > 0.0):
            p = torch.clamp(self.prior_p_vec.to(device), min=1e-6).view(1, 1, -1)
            device_weight = torch.pow(1.0 / p, self.rare_active_alpha) * float(self.rare_active_floor)

        # SmoothL1（Huber）在相对刻度，按设备归约
        delta = self.huber_delta
        resid = torch.abs(pred_n - target_n)
        huber_el = torch.where(resid < delta, 0.5 * resid ** 2, delta * (resid - 0.5 * delta))
        # 在激活期按设备加权（避免“稀有设备低权重”的问题）
        if isinstance(status_seq, torch.Tensor):
            huber_el = huber_el * (1.0 + self.active_boost_weight * device_weight * torch.clamp(status_seq, min=0.0, max=1.0))
        huber_el = torch.where(valid, huber_el, torch.zeros_like(huber_el))
        denom = valid.float().sum(dim=(0, 1)).clamp_min(1.0)  # (K,)
        huber_loss_k = huber_el.sum(dim=(0, 1)) / denom  # (K,)

        # 激活门控相对误差（仅 target_n > 阈值），按设备归约
        rel_thr = self.active_threshold_rel
        act_mask = (target_n > rel_thr) & valid
        rel_err = torch.abs(pred_n - target_n) / (target_n.abs() + 1e-6)
        if isinstance(status_seq, torch.Tensor):
            rel_err = rel_err * (1.0 + self.active_boost_weight * device_weight * torch.clamp(status_seq, min=0.0, max=1.0))
        rel_err = torch.where(act_mask, rel_err, torch.zeros_like(rel_err))
        denom_rel = act_mask.float().sum(dim=(0, 1)).clamp_min(1.0)  # (K,)
        rel_loss_k = rel_err.sum(dim=(0, 1)) / denom_rel  # (K,)

        # 关闭抑制项，按设备归约
        off_pen_k = torch.zeros(K, device=device)
        if isinstance(status_seq, torch.Tensor):
            off_mask = (status_seq <= 0.5) & valid
            off_mag = torch.where(off_mask, torch.relu(pred_seq), torch.zeros_like(pred_seq))
            denom_off = off_mask.float().sum(dim=(0, 1)).clamp_min(1.0)
            off_pen_k = off_mag.sum(dim=(0, 1)) / denom_off

        # 与总体损失保持一致的权重配置
        return huber_loss_k + self.rel_loss_weight * rel_loss_k + self.off_penalty_weight * off_pen_k

    def classification_seq_loss(self, prob_seq: torch.Tensor, status_seq: torch.Tensor,
                                 valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        序列级分类损失：支持 BCE（对概率）或 Focal Loss（对概率）。
        - prob_seq/status_seq: (B, L, K)
        - valid_mask: (B, L, K)
        """
        eps = 1e-6
        prob_soft = torch.clamp(prob_seq, min=eps, max=1 - eps)
        # 二值化 + STE：前向用硬二值，反向用软概率梯度
        if self.classification_hard:
            y_hard = (prob_soft >= float(self.hard_threshold)).float()
            prob = y_hard + prob_soft - prob_soft.detach()
        else:
            prob = prob_soft
        # 为避免 log(0)/log(1) 导致 -inf/NaN，这里对用于损失计算的概率再夹紧
        prob = torch.clamp(prob, min=eps, max=1 - eps)
        y = torch.clamp(status_seq, min=0.0, max=1.0)

        valid = valid_mask if isinstance(valid_mask, torch.Tensor) else (torch.isfinite(prob) & torch.isfinite(y))
        valid = valid.to(torch.bool)

        B, L, K = prob.size(0), prob.size(1), prob.size(2)
        if isinstance(self.pos_weight_vec, torch.Tensor) and self.pos_weight_vec.numel() == K:
            pos_w = self.pos_weight_vec.view(1, 1, -1).to(prob.device)
        else:
            pos_w = torch.ones(1, 1, K, device=prob.device)

        if self.classification_loss_type == "focal":
            # Focal Loss（对概率）：FL = -alpha*(1-pt)^gamma * log(pt)
            # y=1 时 pt=p；y=0 时 pt=1-p
            alpha = self.focal_alpha
            gamma = self.focal_gamma
            pt_pos = prob
            pt_neg = 1.0 - prob
            # 正样本项加入 pos_weight
            loss_pos = -(alpha * pos_w) * (1.0 - pt_pos) ** gamma * torch.log(pt_pos)
            # 稀有负样本增强：当 prior_p 较高（设备大多为开），对 y=0 的项增加按设备权重
            neg_w = torch.ones(1, 1, K, device=prob.device)
            if isinstance(self.prior_p_vec, torch.Tensor) and self.prior_p_vec.numel() == K and (self.rare_neg_alpha > 0.0):
                p = torch.clamp(self.prior_p_vec.to(prob.device), min=0.0, max=1.0).view(1, 1, -1)
                q = torch.clamp(1.0 - p, min=1e-6)  # 负样本占比
                neg_w = torch.pow(1.0 / q, self.rare_neg_alpha) * float(self.rare_neg_floor)
            loss_neg = -(1.0 - alpha) * neg_w * (pt_neg) ** gamma * torch.log(pt_neg)
            el = torch.where(valid, y * loss_pos + (1.0 - y) * loss_neg, torch.zeros_like(prob))
            denom = valid.float().sum().clamp_min(1.0)
            return el.sum() / denom
        else:
            # 标准 BCE（对概率），支持按设备的正样本权重 pos_weight
            loss_pos = -pos_w * y * torch.log(prob)
            loss_neg = -(1.0 - y) * torch.log(1.0 - prob)
            el = torch.where(valid, loss_pos + loss_neg, torch.zeros_like(prob))
            denom = valid.float().sum().clamp_min(1.0)
            return el.sum() / denom

    def classification_seq_loss_per_device(self, prob_seq: torch.Tensor, status_seq: torch.Tensor,
                                           valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """按设备返回序列分类损失 (K,)。支持 focal / BCE，与 classification_seq_loss 一致。
        - 返回值：shape (K,)。
        """
        eps = 1e-6
        prob_soft = torch.clamp(prob_seq, min=eps, max=1 - eps)
        if self.classification_hard:
            y_hard = (prob_soft >= float(self.hard_threshold)).float()
            prob = y_hard + prob_soft - prob_soft.detach()
        else:
            prob = prob_soft
        # 用于对数运算的概率需要避免 0/1 边界
        prob = torch.clamp(prob, min=eps, max=1 - eps)
        y = torch.clamp(status_seq, min=0.0, max=1.0)

        valid = valid_mask if isinstance(valid_mask, torch.Tensor) else (torch.isfinite(prob) & torch.isfinite(y))
        valid = valid.to(torch.bool)

        B, L, K = prob.size(0), prob.size(1), prob.size(2)
        if isinstance(self.pos_weight_vec, torch.Tensor) and self.pos_weight_vec.numel() == K:
            pos_w = self.pos_weight_vec.view(1, 1, -1).to(prob.device)
        else:
            pos_w = torch.ones(1, 1, K, device=prob.device)

        if self.classification_loss_type == "focal":
            alpha = self.focal_alpha
            gamma = self.focal_gamma
            pt_pos = prob
            pt_neg = 1.0 - prob
            loss_pos = -(alpha * pos_w) * (1.0 - pt_pos) ** gamma * torch.log(pt_pos)
            # 稀有负样本增强（按设备维度）
            neg_w = torch.ones(1, 1, K, device=prob.device)
            if isinstance(self.prior_p_vec, torch.Tensor) and self.prior_p_vec.numel() == K and (self.rare_neg_alpha > 0.0):
                p = torch.clamp(self.prior_p_vec.to(prob.device), min=0.0, max=1.0).view(1, 1, -1)
                q = torch.clamp(1.0 - p, min=1e-6)
                neg_w = torch.pow(1.0 / q, self.rare_neg_alpha) * float(self.rare_neg_floor)
            loss_neg = -(1.0 - alpha) * neg_w * (pt_neg) ** gamma * torch.log(pt_neg)
            el = torch.where(valid, y * loss_pos + (1.0 - y) * loss_neg, torch.zeros_like(prob))
        else:
            loss_pos = -pos_w * y * torch.log(prob)
            loss_neg = -(1.0 - y) * torch.log(1.0 - prob)
            el = torch.where(valid, loss_pos + loss_neg, torch.zeros_like(prob))

        denom = valid.float().sum(dim=(0, 1)).clamp_min(1.0)  # (K,)
        return el.sum(dim=(0, 1)) / denom

    def conservation_loss(self, mains_seq: Optional[torch.Tensor], pred_seq: torch.Tensor) -> torch.Tensor:
        """窗口级能量守恒：设备和的时间均值与总功率代表值接近（相对误差）。
        注意：此守恒项不包含未知残差，未知残差由 unknown_residual_loss 处理。
        """
        if not isinstance(mains_seq, torch.Tensor):
            return torch.tensor(0.0, device=pred_seq.device)
        sum_per_t = pred_seq.sum(dim=2)  # (B, L)
        pred_total = sum_per_t.mean(dim=1, keepdim=True)  # (B, 1)
        valid = torch.isfinite(pred_total) & torch.isfinite(mains_seq)
        rel = torch.where(valid, torch.abs(pred_total - mains_seq) / (mains_seq.abs() + 1e-6), torch.zeros_like(mains_seq))
        denom = valid.float().sum().clamp_min(1.0)
        return rel.sum() / denom

    def unknown_residual_loss(self, mains_seq: Optional[torch.Tensor], pred_seq: torch.Tensor,
                               unknown_win: Optional[torch.Tensor]) -> torch.Tensor:
        """未知残差匹配：鼓励未知头拟合主功率中“未由设备解释的部分”，并对过度使用未知施加惩罚。
        - mains_seq: (B, L)
        - pred_seq: (B, L, K)
        - unknown_win: (B, 1) 或 None  -> 窗口级未知功率预测
        返回标量损失： match + penalty
        """
        if not (isinstance(mains_seq, torch.Tensor) and isinstance(unknown_win, torch.Tensor)):
            return torch.tensor(0.0, device=pred_seq.device)
        B, L, K = pred_seq.size(0), pred_seq.size(1), pred_seq.size(2)
        # 设备总功率的窗口均值
        sum_per_t = pred_seq.sum(dim=2)              # (B, L)
        dev_total = sum_per_t.mean(dim=1, keepdim=True)  # (B, 1)
        mains_mean = mains_seq.mean(dim=1, keepdim=True) # (B, 1)
        # 只拟合正的未解释残差（避免负残差导致未知吸收基线）
        residual_target = torch.relu(mains_mean - dev_total)
        # 匹配损失：SmoothL1（窗口级）
        diff = torch.abs(unknown_win - residual_target)
        delta = self.huber_delta
        match = torch.where(diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
        match = match.mean()
        # 使用惩罚抑制滥用未知：L1 强度（越小越好）
        penalty = torch.relu(unknown_win).mean()
        return self.unknown_match_weight * match + float(self.unknown_l1_penalty) * penalty


def create_loss_function(cfg: Dict[str, Any]) -> NILMLoss:
    """创建简洁的 NILM 联合损失对象。"""
    return NILMLoss(cfg or {})