#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : Energformer baseline
#
#################################################################################################################

import math
import torch

from torch import nn
from torch.nn import Module


class FeatureMap(Module):
    """Define the FeatureMap interface."""

    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.
        It is inherited by the subclasses so it is available in all feature
        maps.
        """

        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)

        return inner


class ActivationFunctionFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""

    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self):
        return

    def forward(self, x):
        return self.activation_function(x)


elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)


class SquaredReLU(Module):
    """
    Squared ReLU activation
    """

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Apply ReLU
        x = self.relu(x)
        # Square it
        return x * x


class SquaredGELU(Module):
    """
    ## Squared GELU activation
    """

    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        # Apply GELU
        x = self.gelu(x)
        # Square it
        return x * x


class SpatialDepthWiseConvolution(Module):
    """
    ## Spatial Depth Wise Convolution
    """

    def __init__(self, d_k: int, kernel_size: int = 3):
        """
        * `d_k` is the number of channels in each head
        """
        super().__init__()
        self.kernel_size = kernel_size
        # We use PyTorch's `Conv1d` module.
        # We set the number of groups to be equal to the number of channels so that it does a separate convolution
        # (with different kernels) for each channel.
        # We add padding to both sides and later crop the right most `kernel_size - 1` results
        self.conv = nn.Conv1d(
            in_channels=d_k,
            out_channels=d_k,
            kernel_size=(kernel_size,),
            padding=(kernel_size - 1,),
            groups=d_k,
        )

    def forward(self, x: torch.Tensor):
        """
        'x' has shape '[batch_size, seq_len, heads, d_k]'
        """

        # Get the shape
        batch_size, seq_len, heads, d_k = x.shape
        # Permute to `[batch_size, heads, d_k, seq_len]`
        x = x.permute(0, 2, 3, 1)
        # Change the shape to `[batch_size * heads, d_k, seq_len]`
        x = x.reshape(batch_size * heads, d_k, seq_len)

        # 1D convolution accepts input of the form `[N, channels, sequence]`
        x = self.conv(x)
        # Crop the right most `kernel_size - 1` results since we padded both sides
        x = x[:, :, : -(self.kernel_size - 1)]
        # Reshape to `[batch_size, heads, d_k, seq_len]`
        x = x.reshape(batch_size, heads, d_k, seq_len)
        # Permute to `[batch_size, seq_len, heads, d_k]`
        x = x.permute(0, 3, 1, 2)

        return x


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps):
    Q = q.contiguous().permute(0, 2, 1, 3)
    K = k.contiguous().permute(0, 2, 1, 3)
    V = v.contiguous().permute(0, 2, 1, 3)
    KV = torch.einsum("...sd,...se->...de", K, V)
    Z = 1.0 / torch.einsum("...sd,...d->...s", Q, K.sum(dim=-2) + eps)
    V_new = torch.einsum("...de,...sd,...s->...se", KV, Q, Z)
    return V_new.contiguous().permute(0, 2, 1, 3)


class LinearMultiHeadAttention(nn.Module):
    """
    Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.
    Given the queries, keys and values as Q, K, V instead of computing
        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),
    we make use of a feature map function Φ(.) and perform the following
    computation
        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).
    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.
    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """

    def __init__(self, d_model: int, n_head: int, dropout: float, eps: float = 1e-6):
        super(LinearMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.eps = eps

        assert d_model % n_head == 0, "d_model should be divisible by n_head"
        self.head_dim = d_model // n_head

        # Q, K, V with SpatialDepthWiseConvolution from "Primer: Searching for Efficient Transformers for Language Modeling"
        self.q_proj = nn.Linear(d_model, d_model)
        self.q_sdwc = SpatialDepthWiseConvolution(self.head_dim)
        self.k_proj = nn.Linear(d_model, d_model)
        self.k_sdwc = SpatialDepthWiseConvolution(self.head_dim)
        self.v_proj = nn.Linear(d_model, d_model)
        self.v_sdwc = SpatialDepthWiseConvolution(self.head_dim)

        self.o_proj = nn.Linear(d_model, d_model)

        self.feature_map = elu_feature_map(d_model)

    def forward(self, query, key, value):
        """
        :param query: [batch_size, seq_len, d_model]
        :param key: [batch_size, seq_len, d_model]
        :param value: [batch_size, seq_len, d_model]
        """
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        batch_size, q_len, _ = q.size()
        k_len, v_len = k.size(1), v.size(1)

        # Reshape to [batch_size, seq_len, n_head, head_dim] and apply SpatialDepthWiseConvolution
        q = self.q_sdwc(q.reshape(batch_size, q_len, self.n_head, self.head_dim))
        k = self.k_sdwc(k.reshape(batch_size, k_len, self.n_head, self.head_dim))
        v = self.v_sdwc(v.reshape(batch_size, v_len, self.n_head, self.head_dim))

        self.feature_map.new_feature_map()
        q = self.feature_map.forward_queries(q)
        k = self.feature_map.forward_keys(k)

        V_new = linear_attention(q, k, v, self.eps)

        output = V_new.contiguous().reshape(batch_size, q_len, -1)
        output = self.o_proj(output)

        return output


class ConvFeedForward(nn.Module):
    def __init__(self, d_model, d_cc):
        super(ConvFeedForward, self).__init__()
        """
        According to the paper "instead of a Feed-forward neural network (FNN) we utilize two 1D convolutional layers with kernel set to 1."

        However, Conv1d with kernel size 1 is equivalent to a Linear ... Nevertheless, they also use a squared GELU instead of a standard one
        """
        self.layer1 = nn.Conv1d(in_channels=d_model, out_channels=d_cc, kernel_size=1)
        self.layer2 = nn.Conv1d(in_channels=d_cc, out_channels=d_model, kernel_size=1)
        self.act = SquaredGELU()

    def forward(self, x):
        # Input as [batch, seq_length, dmodel]
        x = self.layer1(x.permute(0, 2, 1))  # Permute as [batch, d_model, seq_length]
        x = self.act(x)
        x = self.layer2(x).permute(0, 2, 1)  # Permute as [batch, seq_length, d_model]
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_head=4, d_ff=None, att_dropout=0.2, ffn_dropout=0.2):
        super().__init__()

        d_ff = d_model * 2 if d_ff is None else d_ff

        self.lmha = LinearMultiHeadAttention(
            d_model=d_model, n_head=n_head, dropout=att_dropout
        )
        self.cffn = ConvFeedForward(d_model=d_model, d_cc=d_ff)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

        self.dropout = nn.Dropout(ffn_dropout)

    def forward(self, x) -> torch.Tensor:
        # x input and output shape [batch, seq_length, d_model] to meet Transformer Convention

        # Attention Block
        new_x = self.lmha(x, x, x)

        # Add and Norm
        x = self.norm1(torch.add(x, new_x))

        # PFFN Block
        new_x = self.cffn(x)

        # Add (dropout) and Norm
        x = self.norm2(torch.add(x, self.dropout(new_x)))

        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class Energformer(nn.Module):
    def __init__(
        self,
        c_in=1,
        c_out=1,
        kernel_embed_size=3,
        n_encoder_layers=4,
        d_model=128,
        n_head=4,
        d_ff=256,
        att_dropout=0.2,
        ffn_dropout=0.2,
    ):
        super().__init__()
        """
        PyTorch implementation of Energformer
        """

        # According to Fig 1 and  Section B "the kernel size for the convolutional layers of our model was set to 3 in both layers and stride was set to 1, with ReLU as activation function."
        self.embedding = nn.Sequential(
            *[
                nn.Conv1d(
                    in_channels=c_in,
                    out_channels=d_model // 2,
                    kernel_size=kernel_embed_size,
                    padding="same",
                ),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=d_model // 2,
                    out_channels=d_model,
                    kernel_size=kernel_embed_size,
                    padding="same",
                ),
                nn.ReLU(),
            ]
        )

        # Fixed Positional Encoding from Attention is all you Need
        self.pe = PositionalEmbedding(d_model, max_len=5000)

        # Transformer Encoder Block
        self.encoder = nn.Sequential(
            *[
                TransformerLayer(
                    d_model=d_model,
                    n_head=n_head,
                    d_ff=d_ff,
                    att_dropout=att_dropout,
                    ffn_dropout=ffn_dropout,
                )
                for _ in range(n_encoder_layers)
            ]
        )

        # According to Fig 1 and  Section B "Between linear layers, we apply a ReLU activation function."
        self.output_ffn = nn.Sequential(
            *[
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, c_out),
            ]
        )

    def forward(self, x):
        # Input as [batch, c_in, seq_length]

        # Forward embedding, permute and add PE
        x = self.embedding(x).permute(0, 2, 1) + self.pe(
            x
        )  # [batch, seq_length, d_model]

        # Transformer Block
        x = self.encoder(x)  # [batch, seq_length, d_model]

        # Final Linear layers and permute
        x = self.output_ffn(x).permute(0, 2, 1)  # [batch, c_out, seq_length]

        return x
