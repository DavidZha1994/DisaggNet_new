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
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]


class CausalMultiHeadAttention(nn.Module):
    """因果多头注意力"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # 注册因果mask
        self.register_buffer('causal_mask', None)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 构建因果mask
        causal_mask = self._get_causal_mask(seq_len, query.device)
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.n_heads, -1, -1)
        
        # Apply additional mask if provided
        if mask is not None:
            additional_mask = mask.unsqueeze(1).unsqueeze(1).expand(-1, self.n_heads, seq_len, -1)
            attn_mask = attn_mask.masked_fill(additional_mask == 0, float('-inf'))
        
        # 使用fused scaled_dot_product_attention（PyTorch 2.0+）
        try:
            context = F.scaled_dot_product_attention(
                Q, K, V, 
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False  # 我们已经提供了mask
            )
        except AttributeError:
            # 回退到手动实现（PyTorch < 2.0）
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            scores = scores + attn_mask
            scores = scores.clamp(min=-50.0, max=50.0)  # 数值保护
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context = torch.matmul(attn_weights, V)
        
        # 重塑输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.w_o(context)
        return output


class TransformerBlock(nn.Module):
    """Transformer块（Pre-LN，移除BatchNorm）"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        
        if causal:
            self.attention = CausalMultiHeadAttention(d_model, n_heads, dropout)
        else:
            self.attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        
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
        # Pre-LN 自注意力
        x_norm = self.norm1(x)
        if self.causal:
            attn_output = self.attention(x_norm, x_norm, x_norm, mask)
        else:
            attn_output, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + self.dropout(attn_output)
        
        # Pre-LN 前馈
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
        self.time_proj: Optional[nn.Linear] = None
        
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
            # (B, T, F) -> (B, F, T) -> Conv1d -> (B, d_model, T) -> (B, T, d_model)
            x = x.transpose(1, 2)
            x = self.input_projection(x)
            x = x.transpose(1, 2)
        else:
            x = self.input_projection(x)
        
        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # 添加时间特征（如果提供）
        if time_features is not None:
            # time_features应该包含hour_sin, hour_cos等
            time_embed = torch.mean(time_features, dim=1, keepdim=True)  # (batch_size, 1, time_dim)
            time_embed = time_embed.expand(-1, seq_len, -1)  # (batch_size, seq_len, time_dim)
            if time_embed.size(-1) != x.size(-1):
                if self.time_proj is None:
                    self.time_proj = nn.Linear(time_embed.size(-1), x.size(-1)).to(x.device)
                time_embed = self.time_proj(time_embed)
            x = x + time_embed
        
        x = self.dropout(x)
        
        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x, mask=mask)
        
        return x


class FreqEncoder(nn.Module):
    """频域编码器（轻量与高效）"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        self.config = config
        self.proj_dim = config.proj_dim
        self.conv_kernel = config.conv1d_kernel
        small_transformer_layers = config.small_transformer_layers
        dropout = config.dropout
        
        # 延迟初始化的Conv1d（以频率bin为通道）
        self.conv1: Optional[nn.Conv1d] = None
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 可选的小Transformer
        if small_transformer_layers > 0:
            self.transformer_layers = nn.ModuleList([
                TransformerBlock(self.proj_dim, 4, self.proj_dim * 2, dropout, causal=False)
                for _ in range(small_transformer_layers)
            ])
        else:
            self.transformer_layers = None
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, freq_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch_size, n_time_frames, n_freq_bins)
        if x is None:
            return None
        
        batch_size, n_time_frames, n_freq_bins = x.size()
        
        # (B, T_f, F_f) -> (B, F_f, T_f)
        x = x.transpose(1, 2)
        
        # 延迟初始化Conv1d以适配动态频率维度
        if self.conv1 is None:
            self.conv1 = nn.Conv1d(in_channels=n_freq_bins, out_channels=self.proj_dim, kernel_size=self.conv_kernel, padding=self.conv_kernel // 2).to(x.device)
        
        conv_out = F.relu(self.conv1(x))  # (B, proj_dim, T_f)
        
        # 应用帧掩码（时间维度）
        if freq_valid_mask is not None:
            m = freq_valid_mask.unsqueeze(1).to(conv_out.dtype)  # (B, 1, T_f)
            conv_out = conv_out * m
        
        # 小Transformer在时间维度上（可选）
        if self.transformer_layers is not None:
            seq = conv_out.transpose(1, 2)  # (B, T_f, proj_dim)
            seq = self.dropout(seq)
            for layer in self.transformer_layers:
                seq = layer(seq, mask=freq_valid_mask)
            # 池化为最终表示（支持掩码）
            if freq_valid_mask is not None:
                m = freq_valid_mask.to(seq.dtype)
                denom = m.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
                freq_repr = (seq * m.unsqueeze(-1)).sum(dim=1) / denom  # (B, proj_dim)
            else:
                freq_repr = torch.mean(seq, dim=1)  # (B, proj_dim)
        else:
            # 直接时间全局池化（支持掩码）
            if freq_valid_mask is not None:
                m = freq_valid_mask.unsqueeze(1).to(conv_out.dtype)
                num = (conv_out * m).sum(dim=-1)
                den = m.sum(dim=-1).clamp_min(1.0)
                freq_repr = num / den  # (B, proj_dim)
            else:
                freq_repr = self.global_pool(conv_out).squeeze(-1)  # (B, proj_dim)
        
        return freq_repr


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, n_aux_features)
        if x is None:
            return None

        batch_size, in_dim = x.size()

        # 延迟初始化线性层以适配输入维度
        if self.fc1 is None:
            self.fc1 = nn.Linear(in_dim, self.hidden_dim).to(x.device)
            self.bn1 = nn.BatchNorm1d(self.hidden_dim).to(x.device)
            self.fc2 = nn.Linear(self.hidden_dim, self.out_dim).to(x.device)

        # 前向传播
        y = self.fc1(x)
        # BatchNorm1d 期望 (N, C)
        y = self.bn1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.norm(y)
        return y


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合（注册频域投射层）"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, gated: bool = True, freq_proj_in: Optional[int] = None, bidirectional: bool = False):
        super().__init__()
        
        self.d_model = d_model
        self.gated = gated
        self.bidirectional = bidirectional
        
        # 交叉注意力：时域作为Query，频域作为Key/Value
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        
        # 双向交叉注意力（可选）
        if bidirectional:
            self.reverse_cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        
        # 频域到时域维度投射（已注册参数）
        if freq_proj_in is not None and freq_proj_in != d_model:
            self.freq_proj = nn.Linear(freq_proj_in, d_model)
        else:
            self.freq_proj = None
        
        # 门控机制
        if gated:
            fusion_input_dim = d_model * 3 if bidirectional else d_model * 2
            self.gate = nn.Sequential(
                nn.Linear(fusion_input_dim, d_model),
                nn.Sigmoid()
            )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, time_repr: torch.Tensor, freq_repr: torch.Tensor) -> torch.Tensor:
        # time_repr: (batch_size, seq_len, d_model)
        # freq_repr: (batch_size, proj_dim) -> 需要扩展到seq_len
        
        if freq_repr is None:
            return time_repr
        
        batch_size, seq_len, d_model = time_repr.size()
        
        # 将频域表示扩展到序列长度
        if freq_repr.dim() == 2:
            freq_repr = freq_repr.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, proj_dim)
        
        # 确保维度匹配（使用已注册层）
        if self.freq_proj is not None:
            freq_repr = self.freq_proj(freq_repr)
        
        # 交叉注意力：时域->频域
        attn_output, _ = self.cross_attention(time_repr, freq_repr, freq_repr)
        
        # 双向交叉注意力（可选）
        if self.bidirectional:
            reverse_attn_output, _ = self.reverse_cross_attention(freq_repr, time_repr, time_repr)
        
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
    """多任务预测头（回归+分类），可选 Unknown/Residual 回归头"""
    
    def __init__(self, d_model: int, n_devices: int, hidden_dim: int = 128, 
                 init_p: Optional[float] = None, enable_film: bool = False, 
                 enable_routing: bool = False, include_unknown: bool = False):
        super().__init__()
        
        self.d_model = d_model
        self.n_devices = n_devices
        self.enable_film = enable_film
        self.enable_routing = enable_routing
        self.include_unknown = include_unknown
        
        # 注意力池化层
        self.attention_pooling = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # 设备嵌入移到输出层作为可学习向量，避免特征泄漏
        # 每个设备头都有独立的可学习嵌入向量
        self.device_embeddings = nn.Embedding(n_devices, d_model)
        
        # FiLM条件化层（可选）
        if enable_film:
            self.film_gamma = nn.Linear(d_model, d_model)
            self.film_beta = nn.Linear(d_model, d_model)
        
        # 轻量级路由机制（可选）
        if enable_routing:
            self.routing_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),  # 融合特征 + 设备嵌入
                nn.Tanh(),
                nn.Linear(d_model, n_devices),
                nn.Softmax(dim=-1)
            )
        
        # 回归头（功率预测）
        self.regression_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
                nn.ReLU()  # 确保功率非负
            ) for _ in range(n_devices)
        ])
        
        # Unknown / Residual 回归头（全局能量未归属部分，非负）
        if self.include_unknown:
            self.unknown_regression_head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
                nn.ReLU()
            )
        else:
            self.unknown_regression_head = None
        
        # 分类头（开关状态）
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(n_devices)
        ])
        
        # 初始化分类头偏置
        if init_p is not None:
            # 支持标量或按设备列表
            if isinstance(init_p, (list, tuple)) or isinstance(init_p, ListConfig):
                for i, head in enumerate(self.classification_heads):
                    p = float(init_p[i]) if i < len(init_p) else float(init_p[-1])
                    p = max(min(p, 1 - 1e-6), 1e-6)
                    final_layer = head[-2]  # Sigmoid前的Linear层
                    nn.init.constant_(final_layer.bias, math.log(p / (1 - p)))
            else:
                p = float(init_p)
                p = max(min(p, 1 - 1e-6), 1e-6)
                for head in self.classification_heads:
                    final_layer = head[-2]
                    nn.init.constant_(final_layer.bias, math.log(p / (1 - p)))
    

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        Args:
            x: 输入特征 (batch_size, seq_len, d_model)
        Returns:
            regression_pred: 回归预测 (batch_size, n_devices)
            classification_pred: 分类预测 (batch_size, n_devices)
            unknown_pred: Unknown 回归预测 (batch_size, 1) 或 None
        """
        batch_size = x.size(0)
        
        # 注意力池化得到全局表示
        pooled_x, _ = self.attention_pooling(x, x, x)  # (batch_size, seq_len, d_model)
        global_repr = pooled_x.mean(dim=1)  # (batch_size, d_model)
        
        # 获取所有设备的嵌入向量
        device_ids = torch.arange(self.n_devices, device=x.device)
        device_embeds = self.device_embeddings(device_ids)  # (n_devices, d_model)
        
        # 扩展全局表示以匹配设备数量
        global_repr_expanded = global_repr.unsqueeze(1).expand(-1, self.n_devices, -1)  # (batch_size, n_devices, d_model)
        device_embeds_expanded = device_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, n_devices, d_model)
        
        # 融合全局特征和设备嵌入
        if self.enable_routing:
            # 轻量级路由：基于全局特征和设备嵌入计算路由权重
            routing_input = torch.cat([global_repr_expanded, device_embeds_expanded], dim=-1)
            routing_weights = self.routing_gate(routing_input)  # (batch_size, n_devices, n_devices)
            
            # 应用路由权重
            device_repr = torch.bmm(routing_weights, device_embeds_expanded)  # (batch_size, n_devices, d_model)
            combined_repr = global_repr_expanded + device_repr
        else:
            # 简单加法融合
            combined_repr = global_repr_expanded + device_embeds_expanded
        
        # FiLM条件化（可选）
        if self.enable_film:
            gamma = self.film_gamma(device_embeds_expanded)
            beta = self.film_beta(device_embeds_expanded)
            combined_repr = gamma * combined_repr + beta
        
        # 多任务预测
        regression_preds = []
        classification_preds = []
        
        for i in range(self.n_devices):
            device_features = combined_repr[:, i, :]  # (batch_size, d_model)
            
            # 回归预测（功率）
            reg_pred = self.regression_heads[i](device_features)  # (batch_size, 1)
            regression_preds.append(reg_pred)
            
            # 分类预测（开关状态）
            cls_pred = self.classification_heads[i](device_features)  # (batch_size, 1)
            classification_preds.append(cls_pred)
        
        # 拼接所有设备的预测结果
        regression_pred = torch.cat(regression_preds, dim=1)  # (batch_size, n_devices)
        classification_pred = torch.cat(classification_preds, dim=1)  # (batch_size, n_devices)
        
        # Unknown / Residual 回归
        unknown_pred: Optional[torch.Tensor]
        if self.unknown_regression_head is not None:
            unknown_pred = self.unknown_regression_head(global_repr)  # (batch_size, 1)
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
        seq_preds: List[torch.Tensor] = []
        for i in range(self.n_devices):
            # 利用每设备的回归头对每个时间步进行预测（Linear支持对最后维度的广播）
            per_dev_seq = self.regression_heads[i](x)  # (B, T, 1)
            seq_preds.append(per_dev_seq.squeeze(-1))  # (B, T)
        seq_regression_pred = torch.stack(seq_preds, dim=-1)  # (B, T, N)
        return seq_regression_pred

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
        pooled_x, _ = self.attention_pooling(x, x, x)
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
        include_unknown = bool(unknown_conf is not None and getattr(unknown_conf, 'enable', False))
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
            include_unknown=include_unknown
        )
        
        # 初始化权重
        self._init_weights()
    
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
                freq_valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        # 时域编码
        time_repr = self.time_encoder(time_features, time_positional, mask=time_valid_mask)

        # 辅助特征编码并融合（使用可学习权重，避免信息重复）
        if self.aux_encoder is not None and aux_features is not None:
            aux_repr = self.aux_encoder(aux_features)  # (batch_size, d_model)
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
                    freq_valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        产生序列→序列回归输出，并同步返回窗口级预测用于约束与评估。
        返回：
        - seq_pred: (B, T, N)
        - regression_pred: (B, N)
        - classification_pred: (B, N)
        - unknown_pred: (B, 1) 或 None
        """
        # 时域编码
        time_repr = self.time_encoder(time_features, time_positional, mask=time_valid_mask)

        # 辅助特征编码并融合
        if self.aux_encoder is not None and aux_features is not None:
            aux_repr = self.aux_encoder(aux_features)
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

        # 序列回归预测
        seq_pred = self.prediction_head.forward_seq(fused_repr)  # (B, T, N)

        # 同步生成窗口级预测用于现有损失与约束
        reg_cls_unknown = self.prediction_head(fused_repr)
        regression_pred, classification_pred = reg_cls_unknown[0], reg_cls_unknown[1]
        unknown_pred = reg_cls_unknown[2] if self.include_unknown else None

        return seq_pred, regression_pred, classification_pred, unknown_pred

    def forward_with_embeddings(self, time_features: torch.Tensor, 
                freq_features: Optional[torch.Tensor] = None,
                time_positional: Optional[torch.Tensor] = None,
                aux_features: Optional[torch.Tensor] = None,
                time_valid_mask: Optional[torch.Tensor] = None,
                freq_valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """与 forward 相同，但返回 (reg, cls, unknown?, embeddings)。"""
        # 时域编码
        time_repr = self.time_encoder(time_features, time_positional, mask=time_valid_mask)

        # 辅助特征编码并融合
        if self.aux_encoder is not None and aux_features is not None:
            aux_repr = self.aux_encoder(aux_features)
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