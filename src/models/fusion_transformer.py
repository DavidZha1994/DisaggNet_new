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
        self.rpe_out_dim = d_model

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
        if time_features is not None:
            fp = time_features.size(-1)
            if self.time_rpe is None:
                self.time_rpe = TimeRPE(in_dim=fp, out_dim=self.rpe_out_dim).to(x.device)
            rpe_seq = self.time_rpe(time_features)
            x = x + rpe_seq

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
    """交叉注意力融合（去除门控，简化为残差相加）"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, freq_proj_in: Optional[int] = None, bidirectional: bool = False):
        super().__init__()
        
        self.d_model = d_model
        self.bidirectional = bidirectional
        
        self.cross_attention = CausalMultiHeadAttention(d_model, n_heads, dropout, causal=False)
        if bidirectional:
            self.reverse_cross_attention = CausalMultiHeadAttention(d_model, n_heads, dropout, causal=False)
        
        if freq_proj_in is not None and freq_proj_in != d_model:
            self.freq_proj = nn.Linear(freq_proj_in, d_model)
        else:
            self.freq_proj = None
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, time_repr: torch.Tensor, freq_repr: torch.Tensor) -> torch.Tensor:
        if freq_repr is None:
            return time_repr
        
        batch_size, seq_len, d_model = time_repr.size()
        if freq_repr.dim() == 2:
            freq_repr = freq_repr.unsqueeze(1).expand(-1, seq_len, -1)
        if self.freq_proj is not None:
            freq_repr = self.freq_proj(freq_repr)
        
        attn_output = self.cross_attention(time_repr, freq_repr, freq_repr)
        if self.bidirectional:
            reverse_attn_output = self.reverse_cross_attention(freq_repr, time_repr, time_repr)
            fused_repr = time_repr + self.dropout((attn_output + reverse_attn_output) / 2)
        else:
            fused_repr = time_repr + self.dropout(attn_output)
        return self.norm(fused_repr)


class MultiTaskHead(nn.Module):
    """预测头（仅回归），去除分类与路由门控"""
    
    def __init__(self, d_model: int, n_devices: int, hidden_dim: int = 128, 
                 include_unknown: bool = False,
                 regression_init_bias: float = 0.0, seq_emb_scale_init: float = 0.1,
                 use_seq_ln: bool = True,
                 reg_use_softplus: bool = False,
                 seq_use_softplus: bool = False,
                 seq_init_bias: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.n_devices = n_devices
        self.enable_film = False
        self.enable_routing = False
        self.include_unknown = include_unknown
        
        # 使用安全注意力池化（非因果）
        self.attention_pooling = CausalMultiHeadAttention(d_model, n_heads=8, dropout=0.1, causal=False)
        
        self.device_embeddings = nn.Embedding(n_devices, d_model)
        
        self.film_gamma = None
        self.film_beta = None
        self.routing_gate = None
        
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
        self.classification_heads = None
        self.seq_regressor = nn.Linear(d_model, n_devices)
        
        try:
            for head in self.regression_heads:
                nn.init.constant_(head[-2].bias, float(regression_init_bias))
            if self.unknown_regression_head is not None:
                nn.init.constant_(self.unknown_regression_head[-2].bias, float(regression_init_bias))
            for head in self.seq_out_heads:
                nn.init.constant_(head[0].bias, float(seq_init_bias))
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
        
        combined_repr = global_repr_expanded + device_embeds_expanded
        
        
        
        regression_preds = []
        for i in range(self.n_devices):
            device_features = combined_repr[:, i, :]
            reg_pred = self.regression_heads[i](device_features)
            regression_preds.append(reg_pred)
        
        regression_pred = torch.cat(regression_preds, dim=1)
        classification_pred = None
        
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
        return self.seq_regressor(x)

    

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

        combined_repr = global_repr_expanded + device_embeds_expanded

        

        # 作为嵌入返回（进行L2归一化以稳定对比/度量学习）
        pred_embeddings = torch.nn.functional.normalize(combined_repr, dim=-1)  # (B, N, D)

        # 多任务预测
        regression_preds = []
        for i in range(self.n_devices):
            device_features = combined_repr[:, i, :]  # (B, D)
            reg_pred = self.regression_heads[i](device_features)
            regression_preds.append(reg_pred)
            pass

        regression_pred = torch.cat(regression_preds, dim=1)
        classification_pred = None

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
            # 辅助特征权重参数（去除门控，保留线性加和）
            self.aux_weight = nn.Parameter(torch.tensor(aux_weight))
        else:
            self.aux_encoder = None
            self.aux_weight = None
        
        # 融合模块
        if config.fusion.type == 'cross_attention' and self.freq_encoder is not None:
            # 支持双向交叉注意力
            bidirectional = getattr(config.fusion, 'bidirectional', False)
            self.fusion = CrossAttentionFusion(
                d_model=config.time_encoder.d_model,
                n_heads=8,
                dropout=0.1,
                freq_proj_in=config.freq_encoder.proj_dim,
                bidirectional=bidirectional
            )
        else:
            self.fusion = None
        
        # 多任务预测头
        # Unknown/Residual 头开关（默认关闭，保持兼容）
        unknown_conf = getattr(getattr(config, 'heads', None), 'unknown', None)
        include_unknown = bool(getattr(unknown_conf, 'enable', True)) if unknown_conf is not None else True
        self.include_unknown = include_unknown

        # 解析激活（兼容 use_softplus 与新的 activation/seq_activation）
        _reg_act = str(getattr(getattr(config.heads, 'regression', None), 'activation', 'softplus')).lower()
        _seq_act = str(getattr(getattr(config.heads, 'regression', None), 'seq_activation', _reg_act)).lower()
        _reg_sp = bool(getattr(getattr(config.heads, 'regression', None), 'use_softplus', _reg_act == 'softplus'))
        _seq_sp = bool(getattr(getattr(config.heads, 'regression', None), 'seq_use_softplus', _seq_act == 'softplus'))

        self.prediction_head = MultiTaskHead(
            d_model=config.time_encoder.d_model,
            n_devices=n_devices,
            hidden_dim=config.heads.regression.hidden,
            include_unknown=include_unknown,
            regression_init_bias=float(getattr(getattr(config.heads, 'regression', None), 'init_bias', 0.0)),
            seq_emb_scale_init=float(getattr(getattr(config.heads, 'regression', None), 'seq_emb_scale', 0.1)),
            use_seq_ln=bool(getattr(getattr(config.heads, 'regression', None), 'use_seq_ln', True)),
            reg_use_softplus=_reg_sp,
            seq_use_softplus=_seq_sp,
            seq_init_bias=float(getattr(getattr(config.heads, 'regression', None), 'seq_init_bias', getattr(getattr(config.heads, 'regression', None), 'init_bias', 0.0)))
        )
        
        # 初始化权重
        self._init_weights()

        pass
    
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
            time_repr = time_repr + self.aux_weight * aux_seq

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
        regression_pred = reg_cls_unknown[0]
        if not self.include_unknown:
            return regression_pred, None, None
        else:
            return regression_pred, None, reg_cls_unknown[2]

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
            time_repr = time_repr + self.aux_weight * aux_seq

        # 频域编码（支持掩码）
        freq_repr = None
        if self.freq_encoder is not None and freq_features is not None:
            freq_repr = self.freq_encoder(freq_features, freq_valid_mask)

        # 融合
        fused_repr = self.fusion(time_repr, freq_repr) if self.fusion is not None else time_repr

        seq_pred = self.prediction_head.forward_seq(fused_repr)
        class_seq_pred = None

        # 同步生成窗口级预测
        reg_cls_unknown = self.prediction_head(fused_repr)
        regression_pred = reg_cls_unknown[0]
        classification_pred = None
        unknown_pred = reg_cls_unknown[2] if self.include_unknown else None
        # 序列级分类
        class_seq_pred = class_seq_pred

        try:
            pass
        except Exception:
            pass

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
            time_repr = time_repr + self.aux_weight * aux_seq

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
            return reg, None, None, emb
        else:
            return reg, None, unk, emb




class ZeroSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sp(x)