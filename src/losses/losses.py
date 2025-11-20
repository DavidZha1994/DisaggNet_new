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

    def regression_seq_loss(self, pred_seq: torch.Tensor, target_seq: torch.Tensor,
                            status_seq: Optional[torch.Tensor], valid_mask: Optional[torch.Tensor],
                            power_scale: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, K = pred_seq.size()
        device = pred_seq.device
        scale = self._ensure_scale(power_scale, device, K)
        pred_n = pred_seq / scale
        target_n = target_seq / scale
        if valid_mask is None:
            valid_mask = torch.ones_like(pred_n, dtype=torch.bool)
        valid_f = valid_mask.float()
        loss_fn = torch.nn.HuberLoss(reduction='none', delta=self.huber_delta)
        base_loss = loss_fn(pred_n, target_n)
        weight_map = torch.ones_like(pred_n)
        if isinstance(status_seq, torch.Tensor):
            weight_map = weight_map + (torch.clamp(status_seq, 0.0, 1.0) * self.active_boost)
            off_mask = (status_seq < 0.5) & (pred_n.abs() > self.active_threshold_rel)
            off_penalty = torch.abs(pred_n) * off_mask.float() * self.off_penalty_weight
        else:
            off_penalty = torch.zeros_like(pred_n)
        total_pixel_loss = (base_loss * weight_map) + off_penalty
        loss_val = (total_pixel_loss * valid_f).sum() / valid_f.sum().clamp(min=1.0)
        return loss_val

    def regression_seq_loss_per_device(self, pred_seq: torch.Tensor, target_seq: torch.Tensor,
                                       status_seq: Optional[torch.Tensor], valid_mask: Optional[torch.Tensor],
                                       power_scale: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, K = pred_seq.size()
        device = pred_seq.device
        scale = self._ensure_scale(power_scale, device, K)
        pred_n = pred_seq / scale
        target_n = target_seq / scale
        valid = valid_mask if isinstance(valid_mask, torch.Tensor) else torch.ones_like(pred_n, dtype=torch.bool)
        valid = valid.to(torch.bool)
        delta = self.huber_delta
        resid = torch.abs(pred_n - target_n)
        huber_el = torch.where(resid < delta, 0.5 * resid ** 2, delta * (resid - 0.5 * delta))
        if isinstance(status_seq, torch.Tensor):
            huber_el = huber_el * (1.0 + torch.clamp(status_seq, 0.0, 1.0) * self.active_boost)
        huber_el = torch.where(valid, huber_el, torch.zeros_like(huber_el))
        denom = valid.float().sum(dim=(0, 1)).clamp(min=1.0)
        huber_loss_k = huber_el.sum(dim=(0, 1)) / denom
        off_pen_k = torch.zeros(K, device=device)
        if isinstance(status_seq, torch.Tensor):
            off_mask = (status_seq < 0.5) & (pred_n.abs() > self.active_threshold_rel) & valid
            off_mag = torch.where(off_mask, torch.abs(pred_n), torch.zeros_like(pred_n))
            denom_off = off_mask.float().sum(dim=(0, 1)).clamp(min=1.0)
            off_pen_k = off_mag.sum(dim=(0, 1)) / denom_off
        return huber_loss_k + self.off_penalty_weight * off_pen_k


    def conservation_loss(self, mains_seq: Optional[torch.Tensor], pred_seq: torch.Tensor,
                          target_seq: Optional[torch.Tensor] = None,
                          power_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mains_seq is None:
            return torch.tensor(0.0, device=pred_seq.device)
        sum_apps = pred_seq.sum(dim=2)
        violation = F.relu(sum_apps - mains_seq)
        loss_violation = violation.mean()
        return loss_violation * self.conservation_weight

    def unknown_residual_loss(self, mains_seq: Optional[torch.Tensor], pred_seq: torch.Tensor,
                               unknown_win: Optional[torch.Tensor], status_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
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


def create_loss_function(cfg: Dict[str, Any]) -> NILMLoss:
    return NILMLoss(cfg or {})