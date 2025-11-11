#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : STNILM baseline
#
#################################################################################################################

import math
import torch

from torch import nn

from src.baselines.nilm.layers.moe import (
    FeedForward,
    SwitchFeedForward,
    clone_module_list,
)
from src.baselines.nilm.bert4nilm import PositionalEmbedding, MultiHeadedAttention


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()

    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, n_experts, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(
            h=attn_heads, d_model=hidden, dropout=dropout
        )
        self.feed_forward = SwitchFeedForward(
            d_model=hidden,
            expert=FeedForward(
                d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
            ),
            n_experts=n_experts,
        )

        self.norm1 = nn.LayerNorm(hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        new_x = self.attention(x, x, x, mask=mask)
        x = torch.add(x, new_x)
        x = self.norm1(x)
        x = self.dropout1(x)

        new_x, counts, route_prob, n_dropped, route_prob_max = self.feed_forward(x)
        x = torch.add(x, new_x)
        x = self.norm2(x)
        x = self.dropout2(x)

        return x, counts, route_prob, n_dropped, route_prob_max


class STNILM(nn.Module):
    def __init__(
        self,
        window_size,
        c_in=1,
        c_out=1,
        n_experts=10,
        dp_rate=0.1,
        weight_moe=0.1,
        criterion=nn.MSELoss(),
    ):
        super().__init__()

        self.latent_len = int(window_size / 2)
        self.dropout_rate = dp_rate

        self.n_experts = n_experts
        self.criterion = criterion
        self.weight_moe = weight_moe
        self.output_size = c_out

        self.hidden = 256
        self.heads = 2
        self.n_layers = 2

        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=self.hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            padding_mode="replicate",
        )
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

        self.position = PositionalEmbedding(
            max_len=self.latent_len, d_model=self.hidden
        )
        self.layer_norm = nn.LayerNorm(self.hidden)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        transformer_layer = TransformerBlock(
            self.hidden, self.heads, self.n_experts, self.hidden * 4, self.dropout_rate
        )
        self.transformer_blocks = clone_module_list(transformer_layer, self.n_layers)

        self.deconv = nn.ConvTranspose1d(
            in_channels=self.hidden,
            out_channels=self.hidden,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.linear1 = nn.Linear(self.hidden, 128)
        self.linear2 = nn.Linear(128, self.output_size)

        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        params = list(self.named_parameters())
        for n, p in params:
            if "layer_norm" in n:
                continue
            else:
                with torch.no_grad():
                    from_l = (
                        1.0 + math.erf(((lower - mean) / std) / math.sqrt(2.0))
                    ) / 2.0
                    to_u = (
                        1.0 + math.erf(((upper - mean) / std) / math.sqrt(2.0))
                    ) / 2.0
                    p.uniform_(2 * from_l - 1, 2 * to_u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.0))
                    p.add_(mean)

    def forward(self, x):
        # Input as B, C, L
        x_token = self.pool(self.conv(x)).permute(0, 2, 1)
        embedding = x_token + self.position(x)
        x = self.dropout(self.layer_norm(embedding))

        # Mixture Of Expert router
        counts, route_prob, n_dropped, route_prob_max = [], [], [], []
        for layer in self.transformer_blocks:
            x, f, p, n_d, p_max = layer(x)
            counts.append(f)
            route_prob.append(p)
            n_dropped.append(n_d)
            route_prob_max.append(p_max)

        x = self.deconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)

        if self.training:
            counts = torch.stack(counts)
            route_prob = torch.stack(route_prob)
            route_prob_max = torch.stack(route_prob_max)

            total = counts.sum(dim=-1, keepdims=True)
            # Fraction of tokens routed to each expert
            route_frac = counts / total
            # Mean routing probability
            route_prob = route_prob / total
            # Load balancing loss
            loss_moe = self.n_experts * (route_frac * route_prob).sum()

            return x.permute(0, 2, 1), loss_moe
        else:
            return x.permute(0, 2, 1)

    def train_one_epoch(self, loader, optimizer, device="cuda"):
        """
        Train STNILM for one epoch
        """
        self.train()
        total_loss = 0

        for seqs, labels, status in loader:
            seqs, labels, status = (
                torch.Tensor(seqs.float()).to(device),
                torch.Tensor(labels.float()).to(device),
                torch.Tensor(status.float()).to(device),
            )

            # Forward model
            optimizer.zero_grad()
            power_logits, loss_moe = self.forward(seqs)

            loss = self.criterion(power_logits, labels)
            loss = loss + self.weight_moe * loss_moe

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss = total_loss / len(loader)

        return total_loss
