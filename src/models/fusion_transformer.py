"""时频域融合Transformer模型"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from omegaconf import DictConfig, ListConfig
import math


class PositionalEncoding(nn.Module):
    """相对位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # 预先构建一个默认长度的PE缓冲区，后续按需动态扩展
        pe = self._build_pe(max_len, device=torch.device('cpu'), dtype=torch.float32)
        self.register_buffer('pe', pe)
    
    def _build_pe(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(0, length, dtype=dtype, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=dtype, device=device) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(length, self.d_model, dtype=dtype, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (length, 1, d_model)
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch_size, d_model)
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            # 动态扩展至匹配输入长度，并保持dtype/device一致
            pe = self._build_pe(seq_len, device=x.device, dtype=x.dtype)
        else:
            pe = self.pe[:seq_len, :].to(dtype=x.dtype, device=x.device)
        return x + pe


class TimeRPE(nn.Module):
    """基于离散时间戳的周期性位置编码（TimeRPE）

    将由分钟/小时/星期/月的 sin/cos 周期特征组成的 `time_positional: (B, T, Fp)`
    通过 `Conv1d(k=1)` 压缩到固定维度，逐时间步与主嵌入拼接。
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # k=1 实际等价于逐时间步的线性投影，但保持Conv1d形式以贴合论文实现
        self.proj = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, time_positional: torch.Tensor) -> torch.Tensor:
        # time_positional: (B, T, Fp)
        b, t, fp = time_positional.size()
        x = time_positional.transpose(1, 2)  # (B, Fp, T)
        y = self.proj(x)                     # (B, out_dim, T)
        y = y.transpose(1, 2)                # (B, T, out_dim)
        return self.dropout(y)


class CausalMultiHeadAttention(nn.Module):
    """因果/通用多头注意力（在MPS上禁用SDPA，使用安全实现）"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # 注册因果mask（仅当causal为True时使用）
        self.register_buffer('causal_mask', None)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            self.causal_mask = mask
        return self.causal_mask

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, q_len, _ = query.size()
        k_len = key.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, q_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, k_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, k_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 构建注意力mask（统一为布尔掩码：True 表示屏蔽）
        attn_mask_bool = None
        if self.causal:
            causal_mask = self._get_causal_mask(q_len, query.device).bool()
            attn_mask_bool = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.n_heads, -1, -1)
        
        if mask is not None:
            # mask 为 key 维度的有效位，转换为无效位布尔掩码
            key_valid = (mask > 0) if mask.dtype != torch.bool else mask
            key_invalid = ~key_valid
            additional_mask = key_invalid.unsqueeze(1).unsqueeze(1).expand(-1, self.n_heads, q_len, -1)
            attn_mask_bool = additional_mask if attn_mask_bool is None else (attn_mask_bool | additional_mask)
        
        # 在MPS上禁用F.scaled_dot_product_attention以避免NDArray断言错误
        use_fused_sdpa = hasattr(F, 'scaled_dot_product_attention') and not torch.backends.mps.is_available()
        
        if use_fused_sdpa:
            context = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask_bool,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # 安全实现（手动softmax路径），兼容MPS
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, q_len, k_len)
            if attn_mask_bool is not None:
                scores = scores.masked_fill(attn_mask_bool, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            # 若整行均为 -inf，softmax 产生 NaN；将其安全置零
            attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
            attn = self.dropout(attn)
            context = torch.matmul(attn, V)  # (B, H, q_len, d_k)
        
        # 合并头
        context = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.w_o(context)


class TransformerBlock(nn.Module):
    """Transformer块（Pre-LN，移除BatchNorm）"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        
        # 统一使用安全注意力实现（在MPS上禁用SDPA）
        self.attention = CausalMultiHeadAttention(d_model, n_heads, dropout, causal=causal)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_output = self.attention(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn_output)
        x_ff = self.feed_forward(self.norm2(x))
        x = x + self.dropout(x_ff)
        return x


class TimeEncoder(nn.Module):
    """时域编码器（PatchTST风格）"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        self.config = config
        d_model = config.d_model
        n_heads = config.n_heads
        num_layers = config.num_layers
        dropout = config.dropout
        
        # 输入嵌入 - 延迟初始化
        self.use_conv_embed = config.input_conv_embed
        self.d_model = d_model
        self.input_projection: Optional[nn.Module] = None
        # TimeRPE 与 TokenStats 相关模块（延迟初始化）
        self.time_rpe: Optional[TimeRPE] = None
        self.rpe_out_dim = d_model // 4
        self.stats_proj: Optional[nn.Linear] = None
        self.stats_out_dim = d_model // 4
        self.concat_proj: Optional[nn.Linear] = None
        self.norm_eps: float = float(getattr(config, 'norm_eps', 1e-6))

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout, causal=config.causal_mask)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, time_features: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch_size, seq_len, n_features)
        batch_size, seq_len, n_features = x.size()

        # 延迟初始化输入投影层
        if self.input_projection is None:
            if self.use_conv_embed:
                # 使用Conv1d跨特征通道进行投影
                self.input_projection = nn.Conv1d(in_channels=n_features, out_channels=self.d_model, kernel_size=3, padding=1).to(x.device)
            else:
                # 使用Linear处理所有特征，保留跨特征关系
                self.input_projection = nn.Linear(n_features, self.d_model).to(x.device)

        # 输入投影
        if self.use_conv_embed:
            x = x.transpose(1, 2)
            x = self.input_projection(x)
            x = x.transpose(1, 2)
        else:
            x = self.input_projection(x)
        x_raw = x

        # —— TokenStats：对投影后的序列做逐通道 z-标准化，并生成统计侧信息 ——
        # 计算均值与方差（时间维度）
        mu = x.mean(dim=1)                             # (B, d_model)
        var = x.var(dim=1, unbiased=False)             # (B, d_model)
        var = torch.clamp(var, min=self.norm_eps)
        std = torch.sqrt(var)
        x = (x - mu.unsqueeze(1)) / (std.unsqueeze(1) + self.norm_eps)
        x = torch.nan_to_num(x)

        # 统计侧信息： [mu, log(var)] -> 线性 -> (B, stats_out_dim) -> 广播到序列
        stats_vec = torch.cat([mu, torch.log(var + self.norm_eps)], dim=-1)  # (B, 2*d_model)
        if self.stats_proj is None:
            self.stats_proj = nn.Linear(2 * self.d_model, self.stats_out_dim).to(x.device)
        stats_seq = self.stats_proj(stats_vec).unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, stats_out_dim)

        # —— TimeRPE：利用真实时间戳的周期位置编码（逐时间步），避免时间维均值广播 ——
        rpe_seq = None
        if time_features is not None:
            fp = time_features.size(-1)
            if self.time_rpe is None:
                self.time_rpe = TimeRPE(in_dim=fp, out_dim=self.rpe_out_dim).to(x.device)
            rpe_seq = self.time_rpe(time_features)  # (B, T, rpe_out_dim)

        # 拼接核心嵌入 + RPE + 统计侧信息 → 映射回 d_model
        parts = [x, stats_seq, x_raw]
        if rpe_seq is not None:
            parts.append(rpe_seq)
        concat = torch.cat(parts, dim=-1)  # (B, T, d_model + stats_out_dim + rpe_out_dim?)
        if self.concat_proj is None:
            self.concat_proj = nn.Linear(concat.size(-1), self.d_model).to(x.device)
        x = self.concat_proj(concat)

        # 位置编码（相对索引）
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)

        x = self.dropout(x)

        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x, mask=mask)

        return x


class FreqEncoder(nn.Module):
    """频域编码器

    输入：`x: (B, T_f, F_f)`，其中 `T_f` 为频域帧数，`F_f` 为频率bin数。
    - 先对频率维做卷积/线性投射到 `proj_dim`
    - 可选：在时间帧上应用 Transformer 层（非因果，支持 `return_sequence` 输出）
    - 输出：`(B, proj_dim)` 或 `(B, T_f, proj_dim)`
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        self.config = config
        self.proj_dim = config.proj_dim
        self.conv_kernel = getattr(config, 'conv1d_kernel', 3)
        dropout = getattr(config, 'dropout', 0.1)
        self.use_transformer = bool(getattr(config, 'use_transformer', False))
        self.num_layers = int(getattr(config, 'num_layers', 2))
        self.n_heads = int(getattr(config, 'n_heads', 4))
        self.return_sequence = bool(getattr(config, 'return_sequence', False))
        
        # 延迟初始化的Conv1d（以频率bin为通道）或线性投射
        self.conv1: Optional[nn.Conv1d] = None
        self.lin_proj: Optional[nn.Linear] = None
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        # 时间帧上的 Transformer（非因果）
        if self.use_transformer:
            self.pos_encoding = PositionalEncoding(self.proj_dim)
            self.tf_layers = nn.ModuleList([
                TransformerBlock(self.proj_dim, self.n_heads, self.proj_dim * 4, dropout, causal=False)
                for _ in range(self.num_layers)
            ])
    
    def forward(self, x: torch.Tensor, freq_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T_f, F_f)
        if x is None:
            return None
        
        batch_size, n_time_frames, n_freq_bins = x.size()
        
        if self.conv1 is None and self.lin_proj is None:
            # 优先使用卷积；若频率bin极小，可切换为线性
            if n_freq_bins >= 3:
                self.conv1 = nn.Conv1d(in_channels=n_freq_bins, out_channels=self.proj_dim, kernel_size=self.conv_kernel, padding=self.conv_kernel // 2).to(x.device)
            else:
                self.lin_proj = nn.Linear(n_freq_bins, self.proj_dim).to(x.device)
        
        if self.conv1 is not None:
            x_c = x.transpose(1, 2)  # (B, F_f, T_f)
            conv_out = F.relu(self.conv1(x_c))  # (B, proj_dim, T_f)
            h = conv_out.transpose(1, 2)  # (B, T_f, proj_dim)
        else:
            h = F.relu(self.lin_proj(x))  # (B, T_f, proj_dim)
        
        # 掩码
        if freq_valid_mask is not None:
            m = freq_valid_mask.unsqueeze(-1).to(h.dtype)  # (B, T_f, 1)
            h = h * m
        
        # Transformer 在时间帧维
        if self.use_transformer:
            q = self.pos_encoding(h.transpose(0, 1)).transpose(0, 1)  # (B, T_f, D)
            for layer in self.tf_layers:
                q = layer(q)
            h = q
        
        if self.return_sequence:
            return h  # (B, T_f, proj_dim)
        
        # 池化为窗口向量
        if freq_valid_mask is not None:
            m = freq_valid_mask.unsqueeze(-1).to(h.dtype)
            num = (h * m).sum(dim=1)
            den = m.sum(dim=1).clamp_min(1.0)
            return num / den  # (B, proj_dim)
        else:
            return h.mean(dim=1)


class AuxEncoder(nn.Module):
    """辅助工程化特征编码器（MLP）

    将非序列的工程化/上下文特征投射到与时域表示相同的维度。
    采用延迟初始化以适配动态输入维度。
    """

    def __init__(self, out_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        # 延迟初始化的层
        self.fc1: Optional[nn.Linear] = None
        self.bn1: Optional[nn.BatchNorm1d] = None
        self.fc2: Optional[nn.Linear] = None
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, aux_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch_size, n_aux_features)
        # aux_valid_mask: (batch_size, n_aux_features) - True表示有效特征
        if x is None:
            return None

        batch_size, in_dim = x.size()

        # 延迟初始化线性层以适配输入维度
        if self.fc1 is None:
            self.fc1 = nn.Linear(in_dim, self.hidden_dim).to(x.device)
            self.bn1 = nn.BatchNorm1d(self.hidden_dim).to(x.device)
            self.fc2 = nn.Linear(self.hidden_dim, self.out_dim).to(x.device)

        # 处理无效值：用0填充，但保持梯度流动
        x_processed = x.clone()
        if aux_valid_mask is not None:
            # 将无效位置设为0，但不阻断梯度
            x_processed = torch.where(aux_valid_mask, x_processed, torch.zeros_like(x_processed))
        else:
            # 如果没有掩码，使用nan_to_num作为后备
            x_processed = torch.nan_to_num(x_processed, nan=0.0, posinf=0.0, neginf=0.0)

        # 前向传播
        y = self.fc1(x_processed)
        # BatchNorm1d 期望 (N, C)
        y = self.bn1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.norm(y)
        
        # 如果有掩码，应用特征级别的权重
        if aux_valid_mask is not None:
            # 计算每个样本的有效特征比例，用于调整输出强度
            valid_ratio = aux_valid_mask.float().mean(dim=1, keepdim=True)  # (batch_size, 1)
            # 避免除零，至少保持10%的强度
            valid_ratio = torch.clamp(valid_ratio, min=0.1)
            y = y * valid_ratio.sqrt()  # 使用平方根避免过度惩罚
        
        return y


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合（在MPS上使用安全注意力实现）"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, gated: bool = True, freq_proj_in: Optional[int] = None, bidirectional: bool = False):
        super().__init__()
        
        self.d_model = d_model
        self.gated = gated
        self.bidirectional = bidirectional
        
        # 交叉注意力：时域作为Query，频域作为Key/Value（非因果）
        self.cross_attention = CausalMultiHeadAttention(d_model, n_heads, dropout, causal=False)
        
        if bidirectional:
            self.reverse_cross_attention = CausalMultiHeadAttention(d_model, n_heads, dropout, causal=False)
        
        if freq_proj_in is not None and freq_proj_in != d_model:
            self.freq_proj = nn.Linear(freq_proj_in, d_model)
        else:
            self.freq_proj = None
        
        if gated:
            fusion_input_dim = d_model * 3 if bidirectional else d_model * 2
            self.gate = nn.Sequential(
                nn.Linear(fusion_input_dim, d_model),
                nn.Sigmoid()
            )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, time_repr: torch.Tensor, freq_repr: torch.Tensor) -> torch.Tensor:
        # time_repr: (B, T, D), freq_repr: (B, D) 或 (B, T, D)
        if freq_repr is None:
            return time_repr
        
        batch_size, seq_len, d_model = time_repr.size()
        
        # 将频域表示扩展到序列长度
        if freq_repr.dim() == 2:
            freq_repr = freq_repr.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 维度投射（如需要）
        if self.freq_proj is not None:
            freq_repr = self.freq_proj(freq_repr)
        
        # 交叉注意力：时域->频域
        attn_output = self.cross_attention(time_repr, freq_repr, freq_repr)
        
        # 双向交叉注意力（可选）
        if self.bidirectional:
            reverse_attn_output = self.reverse_cross_attention(freq_repr, time_repr, time_repr)
        
        # 门控融合
        if self.gated:
            if self.bidirectional:
                concat_repr = torch.cat([time_repr, attn_output, reverse_attn_output], dim=-1)
            else:
                concat_repr = torch.cat([time_repr, attn_output], dim=-1)
            gate_weights = self.gate(concat_repr)
            if self.bidirectional:
                fused_repr = gate_weights * time_repr + (1 - gate_weights) * (attn_output + reverse_attn_output) / 2
            else:
                fused_repr = gate_weights * time_repr + (1 - gate_weights) * attn_output
        else:
            if self.bidirectional:
                fused_repr = time_repr + (attn_output + reverse_attn_output) / 2
            else:
                fused_repr = time_repr + attn_output
        
        # 残差连接和层归一化
        output = self.norm(fused_repr + self.dropout(attn_output))
        return output


class MultiTaskHead(nn.Module):
    """多任务预测头（回归+分类），使用安全注意力池化避免MPS崩溃"""
    
    def __init__(self, d_model: int, n_devices: int, hidden_dim: int = 128, 
                 init_p: Optional[float] = None, enable_film: bool = False, 
                 enable_routing: bool = False, include_unknown: bool = False,
                 classification_enable: bool = True,
                 regression_init_bias: float = 0.0, seq_emb_scale_init: float = 0.1,
                 use_seq_ln: bool = True,
                 reg_use_softplus: bool = True,
                 seq_use_softplus: bool = False):
        super().__init__()
        
        self.d_model = d_model
        self.n_devices = n_devices
        self.enable_film = enable_film
        self.enable_routing = enable_routing
        self.include_unknown = include_unknown
        self.classification_enable = classification_enable
        
        # 使用安全注意力池化（非因果）
        self.attention_pooling = CausalMultiHeadAttention(d_model, n_heads=8, dropout=0.1, causal=False)
        
        self.device_embeddings = nn.Embedding(n_devices, d_model)
        
        if enable_film:
            self.film_gamma = nn.Linear(d_model, d_model)
            self.film_beta = nn.Linear(d_model, d_model)
        
        if enable_routing:
            self.routing_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Tanh(),
                nn.Linear(d_model, n_devices),
                nn.Softmax(dim=-1)
            )
        
        self.regression_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
                ZeroSoftplus() if reg_use_softplus else nn.Identity()
            ) for _ in range(n_devices)
        ])
        
        if self.include_unknown:
            self.unknown_regression_head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
                ZeroSoftplus()
            )
        else:
            self.unknown_regression_head = None
        
        if self.classification_enable:
            self.classification_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                ) for _ in range(n_devices)
            ])
        else:
            self.classification_heads = None
        self.seq_conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.seq_tcn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.seq_ln = nn.LayerNorm(d_model) if use_seq_ln else nn.Identity()
        # 设备嵌入缩放，避免覆盖时间特征
        self.seq_emb_scale = nn.Parameter(torch.tensor(seq_emb_scale_init))

        # 序列Decoder（每设备查询，跨注意力到时序记忆）
        self.dec_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': CausalMultiHeadAttention(d_model, n_heads=4, dropout=0.1, causal=True),
                'cross_attn': CausalMultiHeadAttention(d_model, n_heads=4, dropout=0.1, causal=False),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_model * 4, d_model), nn.Dropout(0.1)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model),
            }) for _ in range(1)
        ])
        self.pos_encoding_seq = PositionalEncoding(d_model)
        self.seq_out_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 1), ZeroSoftplus() if seq_use_softplus else nn.Identity()) for _ in range(n_devices)
        ])
        
        if init_p is not None and self.classification_enable:
            if isinstance(init_p, (list, tuple)) or isinstance(init_p, ListConfig):
                for i, head in enumerate(self.classification_heads):
                    p = float(init_p[i]) if i < len(init_p) else float(init_p[-1])
                    p = max(min(p, 1 - 1e-6), 1e-6)
                    final_layer = head[-2]
                    nn.init.constant_(final_layer.bias, math.log(p / (1 - p)))
            else:
                p = float(init_p)
                p = max(min(p, 1 - 1e-6), 1e-6)
                if self.classification_heads is not None:
                    for head in self.classification_heads:
                        final_layer = head[-2]
                        nn.init.constant_(final_layer.bias, math.log(p / (1 - p)))
        try:
            for head in self.regression_heads:
                nn.init.constant_(head[-2].bias, float(regression_init_bias))
            if self.unknown_regression_head is not None:
                nn.init.constant_(self.unknown_regression_head[-2].bias, float(regression_init_bias))
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size = x.size(0)
        
        pooled_x = self.attention_pooling(x, x, x)  # (batch_size, seq_len, d_model)
        global_repr = pooled_x.mean(dim=1)
        
        device_ids = torch.arange(self.n_devices, device=x.device)
        device_embeds = self.device_embeddings(device_ids)
        
        global_repr_expanded = global_repr.unsqueeze(1).expand(-1, self.n_devices, -1)
        device_embeds_expanded = device_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        if self.enable_routing:
            routing_input = torch.cat([global_repr_expanded, device_embeds_expanded], dim=-1)
            routing_weights = self.routing_gate(routing_input)
            device_repr = torch.bmm(routing_weights, device_embeds_expanded)
            combined_repr = global_repr_expanded + device_repr
        else:
            combined_repr = global_repr_expanded + device_embeds_expanded
        
        if self.enable_film:
            gamma = self.film_gamma(device_embeds_expanded)
            beta = self.film_beta(device_embeds_expanded)
            combined_repr = gamma * combined_repr + beta
        
        regression_preds = []
        classification_preds = []
        for i in range(self.n_devices):
            device_features = combined_repr[:, i, :]
            reg_pred = self.regression_heads[i](device_features)
            regression_preds.append(reg_pred)
            if self.classification_heads is not None:
                cls_pred = self.classification_heads[i](device_features)
                classification_preds.append(cls_pred)
        
        regression_pred = torch.cat(regression_preds, dim=1)
        if self.classification_heads is not None:
            classification_pred = torch.cat(classification_preds, dim=1)
        else:
            classification_pred = torch.zeros(batch_size, self.n_devices, device=x.device)
        
        if self.unknown_regression_head is not None:
            unknown_pred = self.unknown_regression_head(global_repr)
        else:
            unknown_pred = None
        
        return regression_pred, classification_pred, unknown_pred

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        """
        基于时序特征的序列→序列回归预测。
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            seq_regression_pred: (batch_size, seq_len, n_devices)
        """
        y = x.transpose(1,2)
        y = self.seq_conv(y)
        y = self.seq_tcn(y)
        y = y.transpose(1,2)
        y = self.seq_ln(y)  # memory: (B, T, D)

        B, T, D = y.size()
        device_ids = torch.arange(self.n_devices, device=x.device)
        device_embeds = self.device_embeddings(device_ids)  # (N, D)
        seq_preds: List[torch.Tensor] = []
        # 构造查询：位置编码 + 设备嵌入
        base_q = torch.zeros(T, B, D, device=x.device)
        base_q = self.pos_encoding_seq(base_q).transpose(0,1)  # (B, T, D)
        for i in range(self.n_devices):
            emb = device_embeds[i].view(1,1,-1).expand(B, T, -1)
            q = base_q + self.seq_emb_scale * emb
            for layer in self.dec_layers:
                q_norm = layer['norm1'](q)
                q = q + layer['self_attn'](q_norm, q_norm, q_norm)
                q_norm = layer['norm2'](q)
                q = q + layer['cross_attn'](q_norm, y, y)
                q = q + layer['ffn'](layer['norm3'](q))
            out = self.seq_out_heads[i](q)  # (B, T, 1)
            seq_preds.append(out.squeeze(-1))
        seq_regression_pred = torch.stack(seq_preds, dim=-1)  # (B, T, N)
        return seq_regression_pred

    def forward_class_seq(self, x: torch.Tensor) -> torch.Tensor:
        """
        序列级分类预测（逐时间步的开关状态概率）。
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            seq_class_pred: (batch_size, seq_len, n_devices)
        """
        seq_preds: List[torch.Tensor] = []
        device_ids = torch.arange(self.n_devices, device=x.device)
        device_embeds = self.device_embeddings(device_ids)
        for i in range(self.n_devices):
            emb = device_embeds[i].view(1, 1, -1).expand(x.size(0), x.size(1), -1)
            if self.classification_heads is not None:
                per_dev_seq = self.classification_heads[i](x + emb)
                seq_preds.append(per_dev_seq.squeeze(-1))
            else:
                seq_preds.append(torch.zeros(x.size(0), x.size(1), device=x.device))
        seq_class_pred = torch.stack(seq_preds, dim=-1)    # (B, T, N)
        return seq_class_pred

    def forward_with_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        与 forward 一致的计算流程，同时返回每设备的中间特征作为嵌入。
        返回：
        - regression_pred: (B, N)
        - classification_pred: (B, N)
        - unknown_pred: (B, 1) 或 None
        - pred_embeddings: (B, N, D)
        """
        batch_size = x.size(0)

        # 注意力池化得到全局表示
        pooled_x = self.attention_pooling(x, x, x)
        global_repr = pooled_x.mean(dim=1)  # (B, D)

        # 设备嵌入
        device_ids = torch.arange(self.n_devices, device=x.device)
        device_embeds = self.device_embeddings(device_ids)  # (N, D)

        # 扩展
        global_repr_expanded = global_repr.unsqueeze(1).expand(-1, self.n_devices, -1)  # (B, N, D)
        device_embeds_expanded = device_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, D)

        # 融合
        if self.enable_routing:
            routing_input = torch.cat([global_repr_expanded, device_embeds_expanded], dim=-1)
            routing_weights = self.routing_gate(routing_input)  # (B, N, N)
            device_repr = torch.bmm(routing_weights, device_embeds_expanded)  # (B, N, D)
            combined_repr = global_repr_expanded + device_repr
        else:
            combined_repr = global_repr_expanded + device_embeds_expanded

        if self.enable_film:
            gamma = self.film_gamma(device_embeds_expanded)
            beta = self.film_beta(device_embeds_expanded)
            combined_repr = gamma * combined_repr + beta

        # 作为嵌入返回（进行L2归一化以稳定对比/度量学习）
        pred_embeddings = torch.nn.functional.normalize(combined_repr, dim=-1)  # (B, N, D)

        # 多任务预测
        regression_preds = []
        classification_preds = []
        for i in range(self.n_devices):
            device_features = combined_repr[:, i, :]  # (B, D)
            reg_pred = self.regression_heads[i](device_features)
            regression_preds.append(reg_pred)
            cls_pred = self.classification_heads[i](device_features)
            classification_preds.append(cls_pred)

        regression_pred = torch.cat(regression_preds, dim=1)
        classification_pred = torch.cat(classification_preds, dim=1)

        unknown_pred: Optional[torch.Tensor]
        if self.unknown_regression_head is not None:
            unknown_pred = self.unknown_regression_head(global_repr)
        else:
            unknown_pred = None

        return regression_pred, classification_pred, unknown_pred, pred_embeddings




class FusionTransformer(nn.Module):
    """时频域融合Transformer主模型"""
    
    def __init__(self, config: DictConfig, n_devices: int):
        super().__init__()
        
        self.config = config
        self.n_devices = n_devices
        
        # 时域编码器
        self.time_encoder = TimeEncoder(config.time_encoder)
        
        # 频域编码器
        if config.freq_encoder.enable:
            self.freq_encoder = FreqEncoder(config.freq_encoder)
        else:
            self.freq_encoder = None

        # 辅助特征编码器（MLP）
        if hasattr(config, 'aux_encoder') and getattr(config.aux_encoder, 'enable', False):
            aux_hidden = getattr(config.aux_encoder, 'hidden', 128)
            aux_dropout = getattr(config.aux_encoder, 'dropout', 0.1)
            # 添加可配置的辅助特征权重，避免与主干信息重复
            aux_weight = getattr(config.aux_encoder, 'weight', 0.3)  # 默认较低权重
            
            self.aux_encoder = AuxEncoder(
                out_dim=config.time_encoder.d_model,
                hidden_dim=aux_hidden,
                dropout=aux_dropout
            )
            # 门控融合层 - 使用可学习权重而非固定门控
            self.aux_gate = nn.Sequential(
                nn.Linear(config.time_encoder.d_model, config.time_encoder.d_model),
                nn.Sigmoid()
            )
            # 辅助特征权重参数
            self.aux_weight = nn.Parameter(torch.tensor(aux_weight))
        else:
            self.aux_encoder = None
            self.aux_gate = None
            self.aux_weight = None
        
        # 融合模块
        if config.fusion.type == 'cross_attention' and self.freq_encoder is not None:
            # 支持双向交叉注意力
            bidirectional = getattr(config.fusion, 'bidirectional', False)
            self.fusion = CrossAttentionFusion(
                d_model=config.time_encoder.d_model,
                n_heads=8,
                dropout=0.1,
                gated=config.fusion.gated,
                freq_proj_in=config.freq_encoder.proj_dim,
                bidirectional=bidirectional
            )
        else:
            self.fusion = None
        
        # 多任务预测头
        init_p = getattr(getattr(config, 'heads', None), 'classification', None)
        init_p = getattr(init_p, 'init_p', None) if init_p is not None else None
        # Unknown/Residual 头开关（默认关闭，保持兼容）
        unknown_conf = getattr(getattr(config, 'heads', None), 'unknown', None)
        include_unknown = bool(getattr(unknown_conf, 'enable', True)) if unknown_conf is not None else True
        self.include_unknown = include_unknown

        self.prediction_head = MultiTaskHead(
            d_model=config.time_encoder.d_model,
            n_devices=n_devices,
            hidden_dim=config.heads.regression.hidden,
            init_p=init_p,
            enable_film=getattr(getattr(config, 'heads', None), 'conditioning', None) is not None and \
                        getattr(getattr(config.heads, 'conditioning', None), 'enable_film', False),
            enable_routing=getattr(getattr(config, 'heads', None), 'routing', None) is not None and \
                           getattr(getattr(config.heads, 'routing', None), 'enable', False),
            include_unknown=include_unknown,
            classification_enable=bool(getattr(getattr(config.heads, 'classification', None), 'enable', True)),
            regression_init_bias=float(getattr(getattr(config.heads, 'regression', None), 'init_bias', 0.0)),
            seq_emb_scale_init=float(getattr(getattr(config.heads, 'regression', None), 'seq_emb_scale', 0.1)),
            use_seq_ln=bool(getattr(getattr(config.heads, 'regression', None), 'use_seq_ln', True)),
            reg_use_softplus=bool(getattr(getattr(config.heads, 'regression', None), 'use_softplus', True)),
            seq_use_softplus=bool(getattr(getattr(config.heads, 'regression', None), 'seq_use_softplus', False))
        )
        
        # 初始化权重
        self._init_weights()

        # 输出侧 ProjStats，用于对序列回归结果进行反标准化（提升稳态与跨域泛化）
        self.denorm_mu_head = nn.Linear(config.time_encoder.d_model, n_devices)
        self.denorm_logvar_head = nn.Linear(config.time_encoder.d_model, n_devices)
        self.denorm_eps = 1e-6
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, time_features: torch.Tensor, 
                freq_features: Optional[torch.Tensor] = None,
                time_positional: Optional[torch.Tensor] = None,
                aux_features: Optional[torch.Tensor] = None,
                time_valid_mask: Optional[torch.Tensor] = None,
                freq_valid_mask: Optional[torch.Tensor] = None,
                aux_valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        # 时域编码
        time_repr = self.time_encoder(time_features, time_positional, mask=time_valid_mask)

        # 辅助特征编码并融合（使用可学习权重，避免信息重复）
        if self.aux_encoder is not None and aux_features is not None:
            aux_repr = self.aux_encoder(aux_features, aux_valid_mask)  # (batch_size, d_model)
            seq_len = time_repr.size(1)
            aux_seq = aux_repr.unsqueeze(1).expand(-1, seq_len, -1)
            gate = self.aux_gate(aux_seq)
            # 使用可学习权重控制辅助特征的贡献，避免与主干重复
            time_repr = time_repr + self.aux_weight * gate * aux_seq

        # 频域编码（支持掩码）
        freq_repr = None
        if self.freq_encoder is not None and freq_features is not None:
            freq_repr = self.freq_encoder(freq_features, freq_valid_mask)
        
        # 融合
        if self.fusion is not None:
            fused_repr = self.fusion(time_repr, freq_repr)
        else:
            fused_repr = time_repr
        
        # 多任务预测（窗口级）
        reg_cls_unknown = self.prediction_head(fused_repr)
        regression_pred, classification_pred = reg_cls_unknown[0], reg_cls_unknown[1]
        if not self.include_unknown:
            return regression_pred, classification_pred, None
        else:
            return regression_pred, classification_pred, reg_cls_unknown[2]

    def forward_seq(self, time_features: torch.Tensor,
                    freq_features: Optional[torch.Tensor] = None,
                    time_positional: Optional[torch.Tensor] = None,
                    aux_features: Optional[torch.Tensor] = None,
                    time_valid_mask: Optional[torch.Tensor] = None,
                    freq_valid_mask: Optional[torch.Tensor] = None,
                    aux_valid_mask: Optional[torch.Tensor] = None,
                    external_scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        产生序列→序列回归输出，并同步返回窗口级预测用于约束与评估。
        返回：
        - seq_pred: (B, T, N)
        - regression_pred: (B, N)
        - classification_pred: (B, N)
        - unknown_pred: (B, 1) 或 None
        - class_seq_pred: (B, T, N)
        """
        # 时域编码
        time_repr = self.time_encoder(time_features, time_positional, mask=time_valid_mask)

        # 辅助特征编码并融合
        if self.aux_encoder is not None and aux_features is not None:
            aux_repr = self.aux_encoder(aux_features, aux_valid_mask)
            seq_len = time_repr.size(1)
            aux_seq = aux_repr.unsqueeze(1).expand(-1, seq_len, -1)
            gate = self.aux_gate(aux_seq)
            time_repr = time_repr + self.aux_weight * gate * aux_seq

        # 频域编码（支持掩码）
        freq_repr = None
        if self.freq_encoder is not None and freq_features is not None:
            freq_repr = self.freq_encoder(freq_features, freq_valid_mask)

        # 融合
        fused_repr = self.fusion(time_repr, freq_repr) if self.fusion is not None else time_repr

        # 序列回归预测（在标准化域）
        seq_pred_norm = self.prediction_head.forward_seq(fused_repr)  # (B, T, N)
        class_seq_pred = self.prediction_head.forward_class_seq(fused_repr)
        gate = 0.2 + 0.8 * torch.clamp(class_seq_pred, min=0.0, max=1.0)
        seq_pred_norm = seq_pred_norm * gate

        mu_pred = self.denorm_mu_head(fused_repr)
        if external_scale is not None:
            if external_scale.dim() == 1:
                scale = external_scale.view(1, 1, -1).to(fused_repr.device)
            else:
                scale = external_scale.to(fused_repr.device)
            scale = torch.clamp(scale, min=1e-6)
            seq_pred = seq_pred_norm * scale
        else:
            sigma2_pred = F.softplus(self.denorm_logvar_head(fused_repr))
            seq_pred = seq_pred_norm * torch.sqrt(sigma2_pred + self.denorm_eps)

        # 同步生成窗口级预测
        reg_cls_unknown = self.prediction_head(fused_repr)
        regression_pred, classification_pred = reg_cls_unknown[0], reg_cls_unknown[1]
        unknown_pred = reg_cls_unknown[2] if self.include_unknown else None
        # 序列级分类
        class_seq_pred = class_seq_pred

        return seq_pred, regression_pred, classification_pred, unknown_pred, class_seq_pred

    def forward_with_embeddings(self, time_features: torch.Tensor, 
                freq_features: Optional[torch.Tensor] = None,
                time_positional: Optional[torch.Tensor] = None,
                aux_features: Optional[torch.Tensor] = None,
                time_valid_mask: Optional[torch.Tensor] = None,
                freq_valid_mask: Optional[torch.Tensor] = None,
                aux_valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """与 forward 相同，但返回 (reg, cls, unknown?, embeddings)。"""
        # 时域编码
        time_repr = self.time_encoder(time_features, time_positional, mask=time_valid_mask)

        # 辅助特征编码并融合
        if self.aux_encoder is not None and aux_features is not None:
            aux_repr = self.aux_encoder(aux_features, aux_valid_mask)
            seq_len = time_repr.size(1)
            aux_seq = aux_repr.unsqueeze(1).expand(-1, seq_len, -1)
            gate = self.aux_gate(aux_seq)
            time_repr = time_repr + self.aux_weight * gate * aux_seq

        # 频域编码
        freq_repr = None
        if self.freq_encoder is not None and freq_features is not None:
            freq_repr = self.freq_encoder(freq_features, freq_valid_mask)

        # 融合
        if self.fusion is not None:
            fused_repr = self.fusion(time_repr, freq_repr)
        else:
            fused_repr = time_repr

        # 预测并返回嵌入
        reg, cls, unk, emb = self.prediction_head.forward_with_embeddings(fused_repr)
        if not self.include_unknown:
            return reg, cls, None, emb
        else:
            return reg, cls, unk, emb




class TemperatureScaling(nn.Module):
    """温度缩放校准"""
    
    def __init__(self, n_devices: int):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(n_devices))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: (batch_size, n_devices)
        # 返回缩放后的logits，概率由调用方再做sigmoid
        return logits / self.temperatures.unsqueeze(0)
    
    def calibrate(self, logits: torch.Tensor, targets: torch.Tensor, device_idx: int) -> None:
        """在验证集上校准温度参数"""
        from scipy.optimize import minimize_scalar
        
        def nll_loss(temperature):
            scaled_logits = logits[:, device_idx] / temperature
            # 直接对logits使用BCEWithLogits以获得稳定的负对数似然
            loss = F.binary_cross_entropy_with_logits(scaled_logits, targets[:, device_idx])
            return loss.item()
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperatures.data[device_idx] = result.x
        class ZeroSoftplus(nn.Module):
            def __init__(self):
                super().__init__()
                self.sp = nn.Softplus()
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.sp(x)
class ZeroSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sp(x)