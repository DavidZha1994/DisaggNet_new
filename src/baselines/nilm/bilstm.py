#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : BiLSTM baseline
#
#################################################################################################################

import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, window_size, c_in=1, downstreamtask="seq2seq"):
        """
        BiLSTM implementation from Kelly et al. paper
        """
        super(BiLSTM, self).__init__()
        self.window_size = window_size
        self.downstreamtask = downstreamtask

        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=16,
            kernel_size=4,
            dilation=1,
            stride=1,
            bias=True,
            padding="same",
        )
        self.lstm_1 = nn.LSTM(
            input_size=16, hidden_size=64, batch_first=True, bidirectional=True
        )
        self.lstm_2 = nn.LSTM(
            input_size=2 * 64, hidden_size=128, batch_first=True, bidirectional=True
        )
        self.fc_1 = nn.Linear(self.window_size * 128 * 2, 128)

        if self.downstreamtask == "seq2seq":
            self.fc_2 = nn.Linear(128, self.window_size)  # Seq2Seq
        else:
            self.fc_2 = nn.Linear(128, 1)  # Seq2Point
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.conv(x).permute(0, 2, 1)
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        out = self.fc_2(
            self.act(self.fc_1(x.contiguous().view(-1, self.window_size * 256)))
        )

        if self.downstreamtask == "seq2seq":
            return out.unsqueeze(1)
        else:
            return out
