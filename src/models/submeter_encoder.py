import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SubmeterEncoder(nn.Module):
    """
    Training-only encoder that maps sub-meter supervision (states/power) to
    per-device prototype embeddings. This module is NOT used at inference.

    - If target_power is available, a small shared MLP transforms the scalar
      power into an embedding and modulates a learnable device base embedding.
    - If target_power is not available, it falls back to learnable prototypes
      per device.
    """

    def __init__(self, n_devices: int, embed_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.n_devices = n_devices
        self.embed_dim = embed_dim

        # Base prototype per device
        self.device_base = nn.Embedding(n_devices, embed_dim)

        # Shared tiny MLP to transform scalar power to embedding
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # Initialization for stability
        nn.init.normal_(self.device_base.weight, mean=0.0, std=0.02)

    def forward(
        self,
        target_power: Optional[torch.Tensor],
        target_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target_power: (B, N) or None
            target_states: (B, N) binary
        Returns:
            prototypes: (B, N, D)
        """
        B = target_states.size(0)
        N = target_states.size(1)
        assert N == self.n_devices, "Mismatch in number of devices"

        # Base prototypes broadcast to batch
        base = self.device_base.weight.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        # If we have target_power, transform it via MLP and modulate base
        if target_power is not None:
            # Normalize power to a reasonable range
            p = torch.clamp(target_power, min=0.0)
            p = p.unsqueeze(-1)  # (B, N, 1)
            # Flatten to apply shared MLP
            p_flat = p.reshape(-1, 1)
            mod_flat = self.mlp(p_flat)  # (B*N, D)
            mod = mod_flat.view(B, N, self.embed_dim)
            # Only apply modulation when device is ON
            mask = target_states.unsqueeze(-1)  # (B, N, 1)
            prototypes = base + mask * mod
        else:
            prototypes = base

        # L2-normalize for cosine similarity stability
        prototypes = F.normalize(prototypes, dim=-1)
        return prototypes