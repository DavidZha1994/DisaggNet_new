#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : FCN baseline
#
#################################################################################################################

import torch
import torch.nn as nn


# ======================= Simple Fully Convolutional Network (Zhang, AAAI 2018) =======================#
class FCN(nn.Module):
    def __init__(self, window_size, c_in=1, downstreamtask="seq2seq"):
        """
        FCN Pytorch implementation as described in the original paper "Sequence-to-point learning with neural networks for non-intrusive load monitoring".

        Plain Fully Convolutional Neural Network Architecture
        """
        super().__init__()
        self.downstreamtask = downstreamtask

        self.convlayer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=c_in,
                out_channels=30,
                kernel_size=10,
                dilation=1,
                stride=1,
                bias=True,
                padding="same",
            ),
            nn.ReLU(),
        )
        self.convlayer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=30,
                out_channels=30,
                kernel_size=8,
                dilation=1,
                stride=1,
                bias=True,
                padding="same",
            ),
            nn.ReLU(),
        )
        self.convlayer3 = nn.Sequential(
            nn.Conv1d(
                in_channels=30,
                out_channels=40,
                kernel_size=6,
                dilation=1,
                stride=1,
                bias=True,
                padding="same",
            ),
            nn.ReLU(),
        )
        self.convlayer4 = nn.Sequential(
            nn.Conv1d(
                in_channels=40,
                out_channels=50,
                kernel_size=5,
                dilation=1,
                stride=1,
                bias=True,
                padding="same",
            ),
            nn.ReLU(),
        )
        self.convlayer5 = nn.Sequential(
            nn.Conv1d(
                in_channels=50,
                out_channels=50,
                kernel_size=5,
                dilation=1,
                stride=1,
                bias=True,
                padding="same",
            ),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1), nn.LazyLinear(1024), nn.Dropout(0.2)
        )

        if self.downstreamtask == "seq2seq":
            self.out = nn.Linear(1024, window_size)
        else:
            self.out = nn.Linear(1024, 1)

    def forward(self, x) -> torch.Tensor:
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = self.convlayer5(x)
        x = self.fc(x)

        out = self.out(x)

        if self.downstreamtask == "seq2seq":
            return out.unsqueeze(1)
        else:
            return out
