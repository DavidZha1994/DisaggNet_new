import sys
from pathlib import Path

# 让脚本能找到 src.train
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from src.train import NILMLightningModule


# --------------------------------------------------------
# 工具函数：移除 softplus，方便回归头直接学幅值
# --------------------------------------------------------
def _remove_softplus(module: NILMLightningModule):
    ph = module.model.prediction_head
    # 回归头
    for head in ph.regression_heads:
        head[-1] = nn.Identity()
        if hasattr(head[-2], "bias") and head[-2].bias is not None:
            nn.init.constant_(head[-2].bias, 0.0)
    # unknown 头（如果有）
    if getattr(ph, "unknown_regression_head", None) is not None:
        uh = ph.unknown_regression_head
        uh[-1] = nn.Identity()
        if hasattr(uh[-2], "bias") and uh[-2].bias is not None:
            nn.init.constant_(uh[-2].bias, 0.0)
    # 序列输出头（如果有）
    if hasattr(ph, "seq_out_heads"):
        for sh in ph.seq_out_heads:
            sh[-1] = nn.Identity()
            try:
                if hasattr(sh[0], "bias") and sh[0].bias is not None:
                    nn.init.constant_(sh[0].bias, 0.1)
            except Exception:
                pass


# --------------------------------------------------------
# 从 Data/prepared/ukdale 加载一个典型事件窗口
# --------------------------------------------------------
def _load_ukdale_event_window_from_prepared(window_size: int = 256):
    base = Path("Data/prepared/ukdale")
    fold = base / "fold_0"
    if not fold.exists():
        print("[WARN] UKDALE fold_0 not found.")
        return None

    dn_map = base / "device_name_to_id.json"
    if not dn_map.exists():
        print("[WARN] device_name_to_id.json not found.")
        return None

    with open(dn_map, "r", encoding="utf-8") as f:
        name_to_id = json.load(f)
    dev_names = [k for k, _ in sorted(name_to_id.items(), key=lambda kv: kv[1])]

    tr_raw_fp = fold / "train_raw.pt"
    tr_seq_fp = fold / "train_targets_seq.pt"
    tr_stat_fp = fold / "train_status_seq.pt"
    rc_fp = fold / "raw_channel_names.json"
    if not (tr_raw_fp.exists() and tr_seq_fp.exists() and rc_fp.exists()):
        print("[WARN] some train_* files not found.")
        return None

    raw = torch.load(tr_raw_fp)
    ts_obj = torch.load(tr_seq_fp)
    if isinstance(ts_obj, dict):
        targ_all = ts_obj.get("seq")
    else:
        targ_all = ts_obj
    if targ_all is None or not isinstance(targ_all, torch.Tensor):
        print("[WARN] targ_all is None or not Tensor.")
        return None

    try:
        stat = torch.load(tr_stat_fp)
        if isinstance(stat, dict):
            status_all = stat.get("onoff") or stat.get("status")
        else:
            status_all = stat
    except Exception:
        status_all = None

    with open(rc_fp, "r") as f:
        rc = json.load(f)
    try:
        pw_idx = rc.index("P_W")
    except Exception:
        pw_idx = 0

    mains_all = raw[:, :, pw_idx]
    targ_all = torch.nan_to_num(targ_all.float(), nan=0.0).contiguous()
    mains_all = torch.nan_to_num(mains_all.float(), nan=0.0).contiguous()

    # 选一个事件最丰富的窗口（变化最多/总能量最大）
    if status_all is not None and isinstance(status_all, torch.Tensor) and status_all.dim() == 3:
        diff = torch.abs(status_all[:, 1:, :] - status_all[:, :-1, :]).sum(dim=(1, 2))
        idx = int(torch.argmax(diff).item())
    else:
        dev_sum = targ_all.sum(dim=(1, 2))
        idx = int(torch.argmax(dev_sum).item())

    L = int(targ_all.size(1))
    if L < window_size:
        print("[WARN] sequence length < window_size.")
        return None

    tf_pw = mains_all[idx]                      # (L,)
    dP = torch.diff(tf_pw, prepend=tf_pw[:1])   # (L,)
    tf = torch.stack([tf_pw, dP], dim=-1).unsqueeze(0).float()  # (1, L, 2)
    targ = targ_all[idx].unsqueeze(0).float()                   # (1, L, K)

    vals = targ.squeeze(0)
    scales = []
    for k in range(vals.size(1)):
        vk = vals[:, k]
        nz = vk[vk > 1.0]
        if nz.numel() > 0:
            s = torch.quantile(nz, 0.95)
        else:
            s = vk.abs().max()
        s = s.clamp_min(10.0)
        scales.append(s)
    scale = torch.stack(scales)
    scale_t = scale.view(1, 1, -1).float()

    mains_pw = tf_pw.detach().cpu().numpy().astype(np.float32)
    dev_sum = targ.squeeze(0).sum(dim=1).detach().cpu().numpy().astype(np.float32)
    tw = np.arange(L)
    return tf, targ, scale_t, dev_names, mains_pw, dev_sum, tw


# --------------------------------------------------------
# 创建 optimizer / scheduler（带 warmup）
# --------------------------------------------------------
def _make_optimizer(module, opt_name: str, lr: float):
    params = module.model.parameters()
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    elif opt_name == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif opt_name == "radam":
        return torch.optim.RAdam(params, lr=lr)
    elif opt_name == "nadam":
        return torch.optim.NAdam(params, lr=lr)
    else:
        # 默认用 AdamW
        return torch.optim.AdamW(params, lr=lr)


def _make_scheduler(opt, sched: str, lr: float, max_steps: int, use_warmup: bool):
    if sched == "none":
        if use_warmup:
            warm = max(20, int(0.05 * max_steps))
            return torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warm)
        return None

    if sched == "onecycle":
        # OneCycle 自带 warmup，不额外加 LinearLR
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=max_steps,
            pct_start=0.1, anneal_strategy="cos"
        )

    # cosine 类：可加 warmup
    if sched == "cosine":
        if use_warmup:
            warm = max(20, int(0.05 * max_steps))
            sched1 = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warm)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max_steps - warm, eta_min=lr * 0.1
            )
            return torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[sched1, sched2], milestones=[warm]
            )
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max_steps, eta_min=lr * 0.1
            )

    return None


def peak_aware_loss(pred, targ):
    abs_err = torch.abs(pred - targ)
    loss_base = abs_err.mean()
    active = (targ > 10.0).float()
    q = torch.quantile(targ, 0.90, dim=1, keepdim=True)
    peak = (targ >= q).float()
    aw = 5.0
    pw = 10.0
    w = 0.1 + aw * active + pw * peak
    loss_weighted = (abs_err * w).sum() / w.sum().clamp_min(1.0)
    d_pred = pred[:, 1:, :] - pred[:, :-1, :]
    d_targ = targ[:, 1:, :] - targ[:, :-1, :]
    d_err = torch.abs(d_pred - d_targ)
    w_d = (active[:, :-1, :] + peak[:, :-1, :])
    loss_shape = (d_err * w_d).sum() / w_d.sum().clamp_min(1.0)
    return loss_base + 1.0 * loss_weighted + 0.5 * loss_shape


# --------------------------------------------------------
# 单次 overfit 运行：UKDALE 单窗口
# --------------------------------------------------------
def run_overfit_ukdale_simple(
    name: str,
    opt_name: str = "adamw",
    lr: float = 3e-3,
    max_steps: int = 2500,
    sched: str = "cosine",     # "none", "cosine", "onecycle"
    use_warmup: bool = True,
    use_freq: bool = False,
):
    """
    极简 overfit：
    - L1 loss（最干净）
    - 不冻结 encoder
    - 不动态权重、不 shape loss
    - 可选 freq 特征
    - 可 sweep optimizer / lr / scheduler / warmup
    """
    loaded = _load_ukdale_event_window_from_prepared(256)
    if loaded is None:
        print("[ERROR] Failed to load UKDALE window.")
        return

    tf, targ, scale_t, dev_names, mains_pw, dev_sum, tw = loaded

    out_dir = Path(f"outputs/viz/overfit_event_ukdale_sweep_final/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 模型配置 ----------------
    cfg = OmegaConf.create({
        "data": {"device_names": dev_names},
        "model": {
            "d_model": 256,
            "n_heads": 8,
            "num_layers": 4,
            "dropout": 0.0,
            "time_encoder": {
                "d_model": 256,
                "n_heads": 8,
                "num_layers": 4,
                "dropout": 0.0,
                "input_conv_embed": True,
                "causal_mask": True,
            },
            "freq_encoder": {
                "enable": use_freq,
                "proj_dim": 256,
                "use_transformer": False,
                "num_layers": 1,
                "n_heads": 4,
                "return_sequence": False,
            },
            "fusion": {"type": "none", "gated": False},
            "aux_encoder": {"enable": False},
            "heads": {
                "regression": {"hidden": 192, "init_bias": 0.0, "seq_emb_scale": 1.0},
                "conditioning": {"enable_film": False, "dropout": 0.0},
                "classification": {"enable": False},
                "unknown": {"enable": False},
                "routing": {"enable": False},
            },
        },
        "training": {"visualization": {"enable": False}},
        "loss": {
            "regression_weight": 1.0,
            "classification_weight": 0.0,
            "conservation_weight": 0.0,
            "consistency_weight": 0.0,
            "nonneg_penalty_weight": 0.0,
            "normalize_per_device": True,
        },
    })

    module = NILMLightningModule(cfg, {"accelerator": "cpu"}, dev_names)
    _remove_softplus(module)

    # gating 恒为 1（如果模型内部用到了）
    def _always_on(x):
        return torch.ones(x.size(0), x.size(1), len(dev_names), device=x.device)
    if hasattr(module.model.prediction_head, "forward_class_seq"):
        module.model.prediction_head.forward_class_seq = _always_on

    # ---------- freq 特征（可选） ----------
    freq_features = None
    if use_freq:
        tf_pw = tf[0, :, 0].detach()
        n_fft = 128
        hop = 32
        win = torch.hann_window(n_fft)
        stft = torch.stft(tf_pw, n_fft=n_fft, hop_length=hop,
                          window=win, return_complex=True)
        mag = torch.abs(stft).transpose(0, 1).unsqueeze(0).float()
        mag = torch.log1p(mag)
        mag = (mag - mag.mean()) / (mag.std() + 1e-6)
        freq_features = mag

    # ---------- 优化器 & 调度器 ----------
    opt = _make_optimizer(module, opt_name, lr)
    scheduler = _make_scheduler(opt, sched, lr, max_steps, use_warmup)

    # ---------- 训练循环 ----------
    losses = []
    snaps = {}

    for step in range(max_steps):
        opt.zero_grad()

        if use_freq and freq_features is not None:
            seq_pred, _, _, _, _ = module.model.forward_seq(
                tf, external_scale=scale_t, freq_features=freq_features
            )
        else:
            seq_pred, _, _, _, _ = module.model.forward_seq(
                tf, external_scale=scale_t
            )

        loss = peak_aware_loss(seq_pred, targ)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.model.parameters(), 1.0)
        opt.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.item()))

        if step in [300, 600, max_steps - 1]:
            snaps[step] = seq_pred.detach().cpu().numpy()[0]

    # ---------- 输出指标 ----------
    # loss 曲线
    plt.figure(figsize=(8, 3))
    xs = np.arange(len(losses))
    plt.plot(xs, losses, label="loss")
    if len(losses) > 10:
        k = 25
        ker = np.ones(k) / k
        smooth = np.convolve(np.array(losses), ker, mode="same")
        plt.plot(xs, smooth, label="loss_smooth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    # mains vs sum_devices
    targ_np = targ.detach().cpu().numpy()[0]
    x = np.arange(targ_np.shape[0])

    plt.figure(figsize=(10, 4))
    plt.plot(x, mains_pw, label="mains_P_W")
    plt.plot(x, dev_sum, label="sum_devices")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "00_mains_vs_sum_devices.png")
    plt.close()

    # 各设备 target
    for k in range(targ_np.shape[1]):
        plt.figure(figsize=(10, 2.5))
        plt.plot(x, targ_np[:, k], label=f"target_{dev_names[k]}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"01_target_{k}_{dev_names[k]}.png")
        plt.close()

    # 不同 step 的预测
    for step_key in sorted(snaps.keys()):
        pred_np = snaps[step_key]
        for k in range(targ_np.shape[1]):
            plt.figure(figsize=(10, 2.5))
            plt.plot(x, targ_np[:, k], label=f"target_{dev_names[k]}")
            plt.plot(x, pred_np[:, k], label=f"pred_step{step_key}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"02_pred_step{step_key}_{k}_{dev_names[k]}.png")
            plt.close()

        sum_pred = pred_np.sum(axis=1)
        plt.figure(figsize=(10, 4))
        plt.plot(x, mains_pw, label="mains_P_W")
        plt.plot(x, dev_sum, label="sum_targets")
        plt.plot(x, sum_pred, label=f"sum_pred_step{step_key}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"03_mains_vs_sum_step{step_key}.png")
        plt.close()

    # 保存简单 metrics
    final_loss = losses[-1]
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"final_loss={final_loss:.6f}\n")
        f.write(f"first_loss={losses[0]:.6f}\n")
        f.write(f"ratio={losses[0]/max(final_loss,1e-6):.3f}\n")


# --------------------------------------------------------
# sweep：一次跑多种组合
# --------------------------------------------------------
def sweep_overfit_ukdale_simple():
    runs = [
        # name,             opt,    lr,     max_steps, sched,      warmup, use_freq
        ("adamw_cosine_3e-3_warm", "adamw", 3e-3, 2500, "cosine",  True,   False),
        ("adamw_cosine_5e-3_warm", "adamw", 5e-3, 2500, "cosine",  True,   False),
        ("adam_cosine_3e-3_warm",  "adam",  3e-3, 2500, "cosine",  True,   False),
        ("radam_cosine_3e-3_warm", "radam", 3e-3, 2500, "cosine",  True,   False),
        ("adamw_onecycle_3e-3",    "adamw", 3e-3, 2500, "onecycle",False,  False),
        ("adamw_cosine_3e-3_nowm", "adamw", 3e-3, 2500, "cosine",  False,  False),
    ]
    for name, opt_name, lr, steps, sched, warm, use_freq in runs:
        print(f"=== Run: {name} ===")
        run_overfit_ukdale_simple(
            name=name,
            opt_name=opt_name,
            lr=lr,
            max_steps=steps,
            sched=sched,
            use_warmup=warm,
            use_freq=use_freq,
        )


if __name__ == "__main__":
    # 你可以只跑一个组合：
    # run_overfit_ukdale_simple("debug_single", "adamw", 3e-3, 2000, "cosine", True, False)

    # 或者跑完整 sweep：
    sweep_overfit_ukdale_simple()