import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any


RECOMMENDED_LOSS_CONFIGS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "regression_weight": 1.0,
        "classification_weight": 1.0,
        "conservation_weight": 0.1,
        "unknown_weight": 0.0,
        "unknown_match_weight": 1.0,
        "unknown_l1_penalty": 0.1,
        "huber_delta": 1.0,
        "active_threshold_rel": 0.05,
        "off_penalty_weight": 0.5,
        "active_boost": 2.0,
    }
}


class NILMLoss:
    def __init__(self, cfg: Dict[str, Any]):
        self.regression_weight = float(cfg.get("regression_weight", 1.0))
        self.classification_weight = float(cfg.get("classification_weight", 1.0))
        self.conservation_weight = float(cfg.get("conservation_weight", 0.1))
        self.unknown_weight = float(cfg.get("unknown_weight", 0.0))
        self.unknown_match_weight = float(cfg.get("unknown_match_weight", 1.0))
        self.unknown_l1_penalty = float(cfg.get("unknown_l1_penalty", 0.1))
        self.huber_delta = float(cfg.get("huber_delta", 1.0))
        self.active_threshold_rel = float(cfg.get("active_threshold_rel", 0.05))
        self.off_penalty_weight = float(cfg.get("off_penalty_weight", 0.5))
        self.active_boost = float(cfg.get("active_boost", 2.0))
        self.peak_focus_top_p = float(cfg.get("peak_focus_top_p", 0.0))
        self.peak_focus_weight = float(cfg.get("peak_focus_weight", 0.0))
        self.derivative_loss_weight = float(cfg.get("derivative_loss_weight", 0.0))
        self.edge_focus_weight = float(cfg.get("edge_focus_weight", 0.0))
        self.edge_focus_thr_rel = float(cfg.get("edge_focus_thr_rel", 0.03))
        # 设备竞争与稀疏门控（默认关闭，避免影响既有测试）
        self.exclusive_penalty_weight = float(cfg.get("exclusive_penalty_weight", 0.0))
        self.sparsity_weight = float(cfg.get("sparsity_weight", 0.0))
        self.allocation_weight = float(cfg.get("allocation_weight", 0.0))
        self.event_count_weight = float(cfg.get("event_count_weight", 0.0))
        self.active_amplitude_weight = float(cfg.get("active_amplitude_weight", 0.0))
        self.shape_variance_weight = float(cfg.get("shape_variance_weight", 0.0))
        self.per_device_off_scale = None
        self.per_device_event_scale = None
        self.per_device_amplitude_scale = None
        self.per_device_variance_scale = None
        self.exclusive_device_weight = None

    def _ensure_scale(self, power_scale: Optional[torch.Tensor], device: torch.device, K: int) -> torch.Tensor:
        if not isinstance(power_scale, torch.Tensor):
            return torch.ones(1, 1, K, dtype=torch.float32, device=device)
        if power_scale.dim() == 1:
            scale = power_scale.view(1, 1, -1)
        elif power_scale.dim() == 2:
            scale = power_scale.mean(dim=0).view(1, 1, -1)
        else:
            scale = power_scale.view(1, 1, -1)
        return torch.clamp(scale.to(device), min=1.0)

    def _vec(self, v: Optional[torch.Tensor], device: torch.device, K: int, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            if v.dim() == 1 and v.numel() == K:
                return v.view(1, 1, -1).to(device=device, dtype=dtype)
            if v.dim() == 2 and v.size(-1) == K:
                return v.view(1, 1, -1).to(device=device, dtype=dtype)
        return torch.ones(1, 1, K, dtype=dtype, device=device)

    def regression_seq_loss(self, pred_seq: torch.Tensor, target_seq: torch.Tensor,
                            status_seq: Optional[torch.Tensor], valid_mask: Optional[torch.Tensor],
                            power_scale: Optional[torch.Tensor], per_device_boost: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, K = pred_seq.size()
        if valid_mask is None:
            valid_mask = torch.ones_like(pred_seq, dtype=torch.bool)
        valid_f = valid_mask.float()
        loss_fn = torch.nn.HuberLoss(reduction='none', delta=self.huber_delta)
        base_loss = loss_fn(pred_seq, target_seq)
        weight_map = torch.ones_like(pred_seq)
        if self.peak_focus_weight > 0.0 and self.peak_focus_top_p > 0.0:
            try:
                q = torch.quantile(target_seq, 1.0 - self.peak_focus_top_p, dim=1).unsqueeze(1)
                peak_mask = (target_seq > q).float()
                weight_map = weight_map * (1.0 + self.peak_focus_weight * peak_mask)
            except Exception:
                pass
        if isinstance(status_seq, torch.Tensor) and self.edge_focus_weight > 0.0:
            try:
                s = torch.clamp(status_seq, 0.0, 1.0)
                d = torch.abs(s[:, 1:, :] - s[:, :-1, :])
                z = torch.zeros(B, 1, K, device=pred_seq.device, dtype=pred_seq.dtype)
                edges_full = torch.cat([z, d], dim=1)
                em = (edges_full > self.edge_focus_thr_rel).float()
                weight_map = weight_map * (1.0 + self.edge_focus_weight * em)
            except Exception:
                pass
        if isinstance(status_seq, torch.Tensor):
            boost = self.active_boost
            if isinstance(per_device_boost, torch.Tensor):
                if per_device_boost.dim() == 1:
                    boost = per_device_boost.view(1, 1, -1).to(pred_seq.device).to(pred_seq.dtype)
                else:
                    boost = per_device_boost.to(pred_seq.device).to(pred_seq.dtype)
            weight_map = weight_map + (torch.clamp(status_seq, 0.0, 1.0) * boost)
            # 关闭阶段惩罚：改为相对每设备尺度（例如训练集P95）
            try:
                if isinstance(power_scale, torch.Tensor):
                    scale = power_scale.to(pred_seq.device).to(pred_seq.dtype)
                    if scale.dim() == 1:
                        scale = scale.view(1, 1, -1)
                    elif scale.dim() == 2:
                        scale = scale.view(1, 1, -1)
                    pred_norm = torch.abs(pred_seq) / torch.clamp(scale, min=1.0)
                    off_mask = (status_seq < 0.5) & (pred_norm > self.active_threshold_rel)
                else:
                    off_mask = (status_seq < 0.5) & (pred_seq.abs() > self.active_threshold_rel)
                off_penalty = torch.abs(pred_seq) * off_mask.float() * self.off_penalty_weight
            except Exception:
                off_penalty = torch.zeros_like(pred_seq)
        else:
            off_penalty = torch.zeros_like(pred_seq)
        try:
            if isinstance(self.per_device_off_scale, torch.Tensor):
                s = self._vec(self.per_device_off_scale, pred_seq.device, K, pred_seq.dtype)
                off_penalty = off_penalty * s
        except Exception:
            pass
        total_pixel_loss = (base_loss * weight_map) + off_penalty
        der_loss = torch.tensor(0.0, device=pred_seq.device)
        if self.derivative_loss_weight > 0.0:
            try:
                pd = pred_seq[:, 1:, :] - pred_seq[:, :-1, :]
                td = target_seq[:, 1:, :] - target_seq[:, :-1, :]
                vm = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]
                dl = loss_fn(pd, td)
                der_num = (dl * vm.float()).sum()
                der_den = vm.float().sum().clamp(min=1.0)
                der_loss = der_num / der_den
            except Exception:
                der_loss = torch.tensor(0.0, device=pred_seq.device)
        loss_val = (total_pixel_loss * valid_f).sum() / valid_f.sum().clamp(min=1.0)
        return loss_val + self.derivative_loss_weight * der_loss

    def classification_seq_loss(self, pred_seq: torch.Tensor, status_seq: torch.Tensor,
                                valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if not isinstance(status_seq, torch.Tensor):
            return torch.tensor(0.0, device=pred_seq.device)
        if valid_mask is None:
            valid_mask = torch.ones_like(pred_seq, dtype=torch.bool)
        # 统一形状
        if status_seq.shape != pred_seq.shape:
            try:
                status_seq = status_seq.view_as(pred_seq)
            except Exception:
                status_seq = status_seq.expand_as(pred_seq)
        valid_f = valid_mask.to(torch.bool).float()
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        base = loss_fn(pred_seq, status_seq)
        return (base * valid_f).sum() / valid_f.sum().clamp(min=1.0)

    def regression_seq_loss_per_device(self, pred_seq: torch.Tensor, target_seq: torch.Tensor,
                                       status_seq: Optional[torch.Tensor], valid_mask: Optional[torch.Tensor],
                                       power_scale: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, K = pred_seq.size()
        device = pred_seq.device
        valid = valid_mask if isinstance(valid_mask, torch.Tensor) else torch.ones_like(pred_seq, dtype=torch.bool)
        valid = valid.to(torch.bool)
        delta = self.huber_delta
        resid = torch.abs(pred_seq - target_seq)
        huber_el = torch.where(resid < delta, 0.5 * resid ** 2, delta * (resid - 0.5 * delta))
        if isinstance(status_seq, torch.Tensor):
            huber_el = huber_el * (1.0 + torch.clamp(status_seq, 0.0, 1.0) * self.active_boost)
        huber_el = torch.where(valid, huber_el, torch.zeros_like(huber_el))
        denom = valid.float().sum(dim=(0, 1)).clamp(min=1.0)
        huber_loss_k = huber_el.sum(dim=(0, 1)) / denom
        off_pen_k = torch.zeros(K, device=device)
        if isinstance(status_seq, torch.Tensor):
            try:
                if isinstance(power_scale, torch.Tensor):
                    scale = power_scale.to(device).to(pred_seq.dtype)
                    if scale.dim() == 1:
                        scale = scale.view(1, 1, -1)
                    elif scale.dim() == 2:
                        scale = scale.view(1, 1, -1)
                    pred_norm = torch.abs(pred_seq) / torch.clamp(scale, min=1.0)
                    off_mask = (status_seq < 0.5) & (pred_norm > self.active_threshold_rel) & valid
                else:
                    off_mask = (status_seq < 0.5) & (pred_seq.abs() > self.active_threshold_rel) & valid
                off_mag = torch.where(off_mask, torch.abs(pred_seq), torch.zeros_like(pred_seq))
                denom_off = off_mask.float().sum(dim=(0, 1)).clamp(min=1.0)
                off_pen_k = off_mag.sum(dim=(0, 1)) / denom_off
            except Exception:
                off_pen_k = torch.zeros(K, device=device)
        return huber_loss_k + self.off_penalty_weight * off_pen_k

    def conservation_loss(self, mains_seq: Optional[torch.Tensor], pred_seq: torch.Tensor,
                          target_seq: Optional[torch.Tensor] = None,
                          power_scale: Optional[torch.Tensor] = None,
                          valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mains_seq is None:
            return torch.tensor(0.0, device=pred_seq.device)
        sum_apps = pred_seq.sum(dim=2)  # (B, L)
        diff = torch.abs(sum_apps - mains_seq)
        if isinstance(valid_mask, torch.Tensor):
            m = valid_mask.to(torch.float32)
            num = (diff * m).sum()
            den = m.sum().clamp(min=1.0)
            loss_violation = num / den
        else:
            loss_violation = diff.mean()
        return loss_violation * self.conservation_weight

    def unknown_residual_loss(
        self,
        mains_seq: Optional[torch.Tensor],
        pred_seq: torch.Tensor,
        unknown_win: Optional[torch.Tensor],
        status_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not (isinstance(mains_seq, torch.Tensor) and isinstance(unknown_win, torch.Tensor)):
            return torch.tensor(0.0, device=pred_seq.device)
        sum_per_t = pred_seq.sum(dim=2)
        residual_target = torch.relu(mains_seq.mean(dim=1, keepdim=True) - sum_per_t.mean(dim=1, keepdim=True))
        diff = torch.abs(unknown_win - residual_target)
        delta = self.huber_delta
        match = torch.where(diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta)).mean()
        penalty = torch.relu(unknown_win).mean()
        return self.unknown_match_weight * match + float(self.unknown_l1_penalty) * penalty

    def consistency_window_loss(self, seq_pred: torch.Tensor, reg_win: Optional[torch.Tensor],
                                valid_mask: Optional[torch.Tensor], power_scale: Optional[torch.Tensor]) -> torch.Tensor:
        if reg_win is None:
            return torch.tensor(0.0, device=seq_pred.device)
        valid = valid_mask if isinstance(valid_mask, torch.Tensor) else torch.isfinite(seq_pred)
        m = valid.to(torch.float32)
        num = (seq_pred * m).sum(dim=1)
        den = m.sum(dim=1).clamp(min=1.0)
        mean_seq = num / den
        err = torch.abs(mean_seq - reg_win)
        return err.mean()

    def device_exclusive_penalty(self, pred_seq_norm: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        惩罚不同设备在同一时间的重叠预测，鼓励设备间的竞争分配。
        使用 sum(p)^2 - sum(p^2) 来衡量交叉项，避免显式构造 KxK。
        """
        if self.exclusive_penalty_weight <= 0.0:
            return torch.tensor(0.0, device=pred_seq_norm.device)
        m = valid_mask.to(torch.float32) if isinstance(valid_mask, torch.Tensor) else torch.ones_like(pred_seq_norm, dtype=torch.float32)
        p = pred_seq_norm * m
        try:
            K = int(pred_seq_norm.size(-1))
            if isinstance(self.exclusive_device_weight, torch.Tensor):
                w = self._vec(self.exclusive_device_weight, pred_seq_norm.device, K, pred_seq_norm.dtype)
                p = p * w
                m = m * w
        except Exception:
            pass
        sum_p = p.sum(dim=2)            # (B, L)
        sum_p2 = (p ** 2).sum(dim=2)    # (B, L)
        overlap = (sum_p ** 2 - sum_p2) # (B, L)
        den = m.sum().clamp(min=1.0)
        val = overlap.sum() / den
        return val * self.exclusive_penalty_weight

    def sparsity_gate_penalty(self, gate_seq: Optional[torch.Tensor], valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        对门控进行 L1 稀疏约束，鼓励少量设备在任意时刻被激活。
        """
        if self.sparsity_weight <= 0.0 or not isinstance(gate_seq, torch.Tensor):
            return torch.tensor(0.0, device=gate_seq.device if isinstance(gate_seq, torch.Tensor) else torch.device('cpu'))
        if isinstance(valid_mask, torch.Tensor):
            m = valid_mask.to(torch.float32)
            if m.dim() == 2:
                m = m.unsqueeze(-1)
            if m.size(-1) != gate_seq.size(-1):
                m = m.expand(-1, -1, gate_seq.size(-1))
        else:
            m = torch.ones_like(gate_seq, dtype=torch.float32)
        num = (torch.abs(gate_seq) * m).sum()
        den = m.sum().clamp(min=1.0)
        return (num / den) * self.sparsity_weight

    def allocation_distribution_loss(self, gate_seq: Optional[torch.Tensor], target_seq: Optional[torch.Tensor], valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        使门控分配与目标分布一致：KL(target_dist || gate).
        target_dist = target_seq / sum(target_seq, K)；在目标全零或无效时跳过。
        """
        if self.allocation_weight <= 0.0 or not (isinstance(gate_seq, torch.Tensor) and isinstance(target_seq, torch.Tensor)):
            return torch.tensor(0.0, device=gate_seq.device if isinstance(gate_seq, torch.Tensor) else target_seq.device if isinstance(target_seq, torch.Tensor) else torch.device('cpu'))
        eps = 1e-6
        if isinstance(valid_mask, torch.Tensor):
            m = valid_mask.to(torch.float32)
            if m.dim() == 2:
                m = m.unsqueeze(-1)
            if m.size(-1) != target_seq.size(-1):
                m = m.expand(-1, -1, target_seq.size(-1))
        else:
            m = torch.ones_like(target_seq, dtype=torch.float32)
        den = (target_seq * m).sum(dim=-1, keepdim=True)
        tdist = torch.where(den > eps, target_seq / (den + eps), torch.zeros_like(target_seq))
        g = torch.clamp(gate_seq, min=eps, max=1.0)
        kl = (tdist * torch.log((tdist + eps) / g)).sum(dim=-1)  # (B, L)
        num = kl * (den.squeeze(-1) > eps).float()
        mask_time = m[..., 0] if m.dim() == 3 else m
        if mask_time.dim() == 3:
            mask_time = mask_time[..., 0]
        agg = num * mask_time
        val = agg.sum() / mask_time.sum().clamp(min=1.0)
        return val * self.allocation_weight

    def event_count_penalty(self, pred_seq_norm: torch.Tensor, status_seq: Optional[torch.Tensor], valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.event_count_weight <= 0.0 or not isinstance(status_seq, torch.Tensor):
            return torch.tensor(0.0, device=pred_seq_norm.device)
        thr = float(self.active_threshold_rel)
        pred_active = (pred_seq_norm > thr).float()
        true_active = torch.clamp(status_seq, 0.0, 1.0)
        dp = pred_active[:, 1:, :] - pred_active[:, :-1, :]
        dt = true_active[:, 1:, :] - true_active[:, :-1, :]
        pred_events = (dp > 0).float().sum(dim=1)
        true_events = (dt > 0).float().sum(dim=1)
        diff_mat = torch.abs(pred_events - true_events)
        try:
            K = int(pred_seq_norm.size(-1))
            if isinstance(self.per_device_event_scale, torch.Tensor):
                w = self._vec(self.per_device_event_scale, pred_seq_norm.device, K, pred_seq_norm.dtype).view(1, -1)
                num = (diff_mat * w).sum()
                den = w.sum().clamp(min=1.0) * max(int(diff_mat.size(0)), 1)
                diff = num / den
            else:
                diff = diff_mat.mean()
        except Exception:
            diff = diff_mat.mean()
        L = max(int(pred_seq_norm.size(1)), 1)
        return (diff / float(L)) * self.event_count_weight

    def active_amplitude_loss(self, pred_seq_norm: torch.Tensor, target_seq_norm: torch.Tensor, status_seq: Optional[torch.Tensor], valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.active_amplitude_weight <= 0.0 or not isinstance(status_seq, torch.Tensor):
            return torch.tensor(0.0, device=pred_seq_norm.device)
        m = valid_mask.to(torch.float32) if isinstance(valid_mask, torch.Tensor) else torch.ones_like(pred_seq_norm, dtype=torch.float32)
        a = torch.clamp(status_seq, 0.0, 1.0)
        w = m * a
        num = w.sum(dim=1).clamp(min=1.0)
        pred_mean = (pred_seq_norm * w).sum(dim=1) / num
        true_mean = (target_seq_norm * w).sum(dim=1) / num
        diff = torch.abs(pred_mean - true_mean)
        delta = float(self.huber_delta)
        hub_el = torch.where(diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
        try:
            K = int(pred_seq_norm.size(-1))
            if isinstance(self.per_device_amplitude_scale, torch.Tensor):
                w_k = self._vec(self.per_device_amplitude_scale, pred_seq_norm.device, K, pred_seq_norm.dtype).view(1, -1)
                num = (hub_el * w_k).sum()
                den = w_k.sum().clamp(min=1.0) * max(int(hub_el.size(0)), 1)
                hub = num / den
            else:
                hub = hub_el.mean()
        except Exception:
            hub = hub_el.mean()
        return hub * self.active_amplitude_weight

    def shape_variance_loss(self, pred_seq_norm: torch.Tensor, target_seq_norm: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        匹配时序形状的方差，抑制输出被压平：鼓励预测的时间方差接近目标。
        """
        if self.shape_variance_weight <= 0.0:
            return torch.tensor(0.0, device=pred_seq_norm.device)
        m = valid_mask.to(torch.float32) if isinstance(valid_mask, torch.Tensor) else torch.ones_like(pred_seq_norm, dtype=torch.float32)
        # 时间维归一化权重
        den = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        pm = (pred_seq_norm * m).sum(dim=1, keepdim=True) / den
        tm = (target_seq_norm * m).sum(dim=1, keepdim=True) / den
        pvar = ((pred_seq_norm - pm) ** 2 * m).sum(dim=1) / den.squeeze(1)
        tvar = ((target_seq_norm - tm) ** 2 * m).sum(dim=1) / den.squeeze(1)
        diff = torch.abs(pvar - tvar)
        delta = float(self.huber_delta)
        hub_el = torch.where(diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
        try:
            K = int(pred_seq_norm.size(-1))
            if isinstance(self.per_device_variance_scale, torch.Tensor):
                w_k = self._vec(self.per_device_variance_scale, pred_seq_norm.device, K, pred_seq_norm.dtype).view(1, -1)
                num = (hub_el * w_k).sum()
                den = w_k.sum().clamp(min=1.0) * max(int(hub_el.size(0)), 1)
                hub = num / den
            else:
                hub = hub_el.mean()
        except Exception:
            hub = hub_el.mean()
        return hub * self.shape_variance_weight


def create_loss_function(cfg: Dict[str, Any]) -> NILMLoss:
    return NILMLoss(cfg or {})
