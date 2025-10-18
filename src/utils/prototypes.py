import torch
import torch.nn as nn
from typing import Optional, Tuple


class PrototypeLibrary(nn.Module):
    """设备原型统计与 Mahalanobis 距离计算。

    - 以流式统计每个设备的均值和协方差（通过 sum 与 sum_outer）。
    - 提供批量 Mahalanobis 距离计算，带对角正则以稳定逆矩阵。
    """

    def __init__(self, n_devices: int, embed_dim: int, eps: float = 1e-3, min_count: int = 10):
        super().__init__()
        self.n_devices = n_devices
        self.embed_dim = embed_dim
        self.eps = eps
        self.min_count = min_count

        # 统计量
        self.register_buffer('count', torch.zeros(n_devices, dtype=torch.long))
        self.register_buffer('sum_vec', torch.zeros(n_devices, embed_dim))
        self.register_buffer('sum_outer', torch.zeros(n_devices, embed_dim, embed_dim))

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, states: Optional[torch.Tensor] = None) -> None:
        """更新原型统计。
        embeddings: (B, N, D)
        states: (B, N) 开关状态，若提供，则仅在 state>0 时更新。
        """
        assert embeddings.dim() == 3, "embeddings 形状应为 (B, N, D)"
        B, N, D = embeddings.shape
        assert N == self.n_devices and D == self.embed_dim, "embeddings 维度与库初始化不匹配"

        device = embeddings.device
        # 展平为样本列表 (B*N, D)
        flat = embeddings.reshape(B * N, D)
        if states is not None:
            mask = (states > 0).reshape(B * N)
        else:
            mask = torch.ones(B * N, dtype=torch.bool, device=device)

        # 遍历设备（避免大矩阵逐项索引的带宽浪费）
        for i in range(self.n_devices):
            idx = torch.arange(B, device=device) * N + i
            sel = mask[idx]
            if sel.any():
                samples = flat[idx][sel]  # (S, D)
                s = samples.sum(dim=0)  # (D)
                so = samples.t().mm(samples)  # (D, D)
                self.sum_vec[i] += s
                self.sum_outer[i] += so
                self.count[i] += samples.shape[0]

    def get_prototype(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """返回第 i 个设备的 (mean, cov, count)。"""
        c = int(self.count[i].item())
        if c <= 0:
            mean = torch.zeros(self.embed_dim, device=self.sum_vec.device)
            cov = torch.eye(self.embed_dim, device=self.sum_vec.device) * self.eps
            return mean, cov, c
        mean = self.sum_vec[i] / float(c)
        cov = self.sum_outer[i] / float(c) - torch.ger(mean, mean)
        # 数值稳定性与正定性修复
        cov = cov + torch.eye(self.embed_dim, device=cov.device) * self.eps
        return mean, cov, c

    def is_ready(self, i: int) -> bool:
        return int(self.count[i].item()) >= self.min_count

    @torch.no_grad()
    def mahalanobis(self, embeddings: torch.Tensor) -> torch.Tensor:
        """计算批量 Mahalanobis 距离。
        embeddings: (B, N, D)
        返回: (B, N) 距离矩阵；若某设备统计不足，返回该列为 0。
        """
        assert embeddings.dim() == 3
        B, N, D = embeddings.shape
        assert N == self.n_devices and D == self.embed_dim

        device = embeddings.device
        distances = torch.zeros(B, N, device=device)

        for i in range(self.n_devices):
            mean, cov, c = self.get_prototype(i)
            if c < self.min_count:
                continue
            try:
                # 逆协方差（带正则）
                inv_cov = torch.linalg.inv(cov)
            except RuntimeError:
                # 回退到对角近似
                inv_cov = torch.diag(1.0 / torch.diag(cov))

            diff = embeddings[:, i, :] - mean  # (B, D)
            # d^2 = x^T inv_cov x
            # 利用批量矩阵乘以向量: (B, D) @ (D, D) -> (B, D)
            tmp = diff @ inv_cov  # (B, D)
            d2 = (tmp * diff).sum(dim=-1)  # (B)
            distances[:, i] = torch.sqrt(torch.relu(d2))

        return distances