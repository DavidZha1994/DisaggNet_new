#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : TSILNet baseline
#
#################################################################################################################

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class ResidualSelfAttention(nn.Module):
    def __init__(self, dim):
        super(ResidualSelfAttention, self).__init__()
        self.SA = torch.nn.MultiheadAttention(dim, num_heads=1, batch_first=True)

    def forward(self, input):
        x, att_weights = self.SA(
            input.permute(0, 2, 1), input.permute(0, 2, 1), input.permute(0, 2, 1)
        )
        return input + x.permute(0, 2, 1), att_weights


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.rsa1 = ResidualSelfAttention(n_outputs)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.rsa2 = ResidualSelfAttention(n_outputs)

        self.net1 = nn.Sequential(self.conv1, self.chomp1, self.act1, self.dropout1)
        self.net2 = nn.Sequential(self.conv2, self.chomp2, self.act2, self.dropout2)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.gelu = nn.GELU()

    def forward(self, input):
        x = self.net1(input)
        x, att1 = self.rsa1(x)

        x = self.net2(x)
        x, att2 = self.rsa2(x)

        # Calculate the enhanced residual
        res = input if self.downsample is None else self.downsample(input)
        # att_weights_combined = att1.mean(dim=1) + att2.mean(dim=1)  # Combine attention weights
        # enhanced_residual = res * att_weights_combined.unsqueeze(1)

        # Add enhanced residual and apply activation
        x = x + res
        return self.gelu(x)


class TSTCN_Block(nn.Module):
    def __init__(
        self, in_channels, num_channels=[4, 16, 64], kernel_size=5, dropout=0.2
    ):
        super(TSTCN_Block, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels_block = in_channels if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels_block,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalConv1D(nn.Module):
    """
    A 1D causal convolution layer.
    Ensures that the convolution output at time t is only influenced by inputs from time t and earlier.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution. Default is 1.
            dilation (int): Dilation factor. Default is 1.
        """
        super(CausalConv1D, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Padding to ensure causality
        self.padding = (kernel_size - 1) * dilation

        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor after causal convolution.
        """
        # Perform the convolution
        output = self.conv(x)

        # Remove the extra padding on the right to ensure causality
        if self.padding != 0:
            output = output[:, :, : -self.padding]

        return output


class IECA(nn.Module):
    """
    Improved Efficient Channel Attention (IECA) Mechanism with Dilated Causal Convolution.
    """

    def __init__(self, kernel_size=3, dilation=8):
        super(IECA, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dilated_conv = CausalConv1D(
            1, 1, kernel_size=kernel_size, dilation=dilation
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling to summarize each channel
        avg_pooled = self.global_avg_pool(x)  # Shape: [B, C, 1]

        # Apply dilated causal convolution
        conv_out = self.dilated_conv(avg_pooled.transpose(1, 2))  # Shape: [B, 1, C]
        attention_weights = self.sigmoid(conv_out.transpose(1, 2))  # Shape: [B, C, 1]

        # Multiply attention weights with the original input
        out = x * attention_weights
        return out


class IECA_LSTM(nn.Module):
    """
    IECA-LSTM Block: Combines Improved Efficient Channel Attention (IECA) and LSTM.
    """

    def __init__(self, input_channels, dilation=8, hidden_size=[128, 256], dropout=0.2):
        super(IECA_LSTM, self).__init__()
        self.ieca = IECA(dilation=dilation)
        self.lstm1 = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size[0],
            num_layers=1,
            batch_first=True,
            dropout=dropout,
        )

        self.lstm2 = nn.LSTM(
            input_size=hidden_size[0],
            hidden_size=hidden_size[1],
            num_layers=1,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x):
        # Input shape: [B, C, T]
        out = self.ieca(x)  # Shape: [B, C, T]
        x = x + out  # Skip connection as shown in Fig 1
        x = x.permute(0, 2, 1)  # Shape: [B, T, C] for LSTM
        x, _ = self.lstm1(x)  # Shape: [B, T, H]
        x, _ = self.lstm2(x)
        return x.permute(0, 2, 1)


class TSILNet(nn.Module):
    """
    TSILNet: A hybrid model combining TSTCN and IECA-LSTM for sequence modeling.
    """

    def __init__(
        self,
        c_in,
        window_size=480,
        downstreamtask="seq2seq",
        tcn_channels=[4, 16, 64],
        tcn_kernel_size=5,
        tcn_dropout=0.2,
        lstm_hidden_sizes=[128, 256],
        lstm_dropout=0.2,
        dilation=8,
        head_ffn_dim=512,
        head_dropout=0.2,
    ):
        super(TSILNet, self).__init__()

        self.downstreamtask = downstreamtask

        # TSTCN Block
        self.tstcn = TSTCN_Block(
            in_channels=c_in,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
        )

        # IECA-LSTM Block
        self.ieca_lstm = IECA_LSTM(
            input_channels=tcn_channels[-1],
            dilation=dilation,
            hidden_size=lstm_hidden_sizes,
            dropout=lstm_dropout,
        )

        # Fully connected layers for regression
        self.fc1 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(lstm_hidden_sizes[-1] * window_size, head_ffn_dim),
            nn.Tanh(),
            nn.Dropout(head_dropout),
        )

        if self.downstreamtask == "seq2seq":
            self.fc2 = nn.Linear(head_ffn_dim, window_size)  # Seq2Seq
        else:
            self.fc2 = nn.Linear(head_ffn_dim, 1)  # Seq2Point

    def forward(self, x):
        # Input shape: [B, C, T]

        # TSTCN Block
        x = self.tstcn(x)  # Shape: [B, C_TCN, T]

        # IECA-LSTM Block
        x = self.ieca_lstm(x)  # Shape: [B, C_LSTM, T]

        # Fully connected layers
        x = self.fc1(x)  # Shape: [B, 512]
        x = self.fc2(x)

        if self.downstreamtask == "seq2seq":
            return x.unsqueeze(1)
        else:
            return x
