import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))

        stdv = 1.0 / math.sqrt(hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

        self.noise2hidden = nn.Linear(100, hidden_size)
        self.noise2cell = nn.Linear(100, hidden_size)

    def init_hidden(self, noise):
        self.hidden = self.noise2hidden(noise)
        self.cell = self.noise2cell(noise)

    def forward(self, x):
        hidden = self.hidden
        cell = self.cell

        gates = x @ self.W + hidden @ self.U + self.bias
        i, f, g, o = (
            torch.sigmoid(gates[:, : self.hidden_size]),
            torch.sigmoid(gates[:, self.hidden_size : self.hidden_size * 2]),
            torch.tanh(gates[:, self.hidden_size * 2 : self.hidden_size * 3]),
            torch.sigmoid(gates[:, self.hidden_size * 3 :]),
        )
        cell = f * cell + i * g
        hidden = o * torch.tanh(cell)

        self.hidden = hidden
        self.cell = cell

        return hidden, cell


class Affine(nn.Module):
    def __init__(self, num_features, num_hidden):
        super(Affine, self).__init__()
        self.mlp_gamma = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(num_hidden, num_hidden)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(num_hidden, num_features)),
                ]
            )
        )
        self.mlp_beta = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(num_hidden, num_hidden)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(num_hidden, num_features)),
                ]
            )
        )

        nn.init.zeros_(self.mlp_gamma.linear2.weight)
        nn.init.ones_(self.mlp_gamma.linear2.bias)
        nn.init.zeros_(self.mlp_beta.linear2.weight)
        nn.init.zeros_(self.mlp_beta.linear2.bias)

    def forward(self, x, lstm_output):
        weight = self.mlp_gamma(lstm_output)
        bias = self.mlp_beta(lstm_output)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        weight = weight.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand_as(x)

        return weight * x + bias


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, lstm):
        super(GeneratorBlock, self).__init__()

        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, 1)

        self.lstm = lstm

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.affine0 = Affine(in_channels, lstm.hidden_size)
        self.affine1 = Affine(in_channels, lstm.hidden_size)
        self.affine2 = Affine(out_channels, lstm.hidden_size)
        self.affine3 = Affine(out_channels, lstm.hidden_size)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, lstm_input=None):
        lstm_output0, _ = self.lstm(lstm_input)
        hidden = self.affine0(x, lstm_output0)
        hidden = F.leaky_relu(hidden, 0.2, inplace=True)

        lstm_output1, _ = self.lstm(lstm_input)
        hidden = self.affine1(hidden, lstm_output1)
        hidden = F.leaky_relu(hidden, 0.2, inplace=True)

        hidden = self.conv1(hidden)

        lstm_output2, _ = self.lstm(lstm_input)
        hidden = self.affine2(hidden, lstm_output2)
        hidden = F.leaky_relu(hidden, 0.2, inplace=True)

        lstm_output3, _ = self.lstm(lstm_input)
        hidden = self.affine3(hidden, lstm_output3)
        hidden = F.leaky_relu(hidden, 0.2, inplace=True)

        return self.conv2(hidden)


class Generator(nn.Module):
    def __init__(self, ndf, noise_dim, lstm_input_dim, lstm_hidden_dim):
        super(Generator, self).__init__()
        self.ndf = ndf
        self.noize_dim = noise_dim
        self.lstm = CustomLSTM(lstm_input_dim, lstm_hidden_dim)
        self.input_layer = nn.Linear(noise_dim, ndf * 8 * 4 * 4)
        # self.blocks = nn.ModuleList(
        #     [
        #         GeneratorBlock(ndf * 8, ndf * 8, self.lstm),
        #         GeneratorBlock(ndf * 8, ndf * 8, self.lstm),
        #         GeneratorBlock(ndf * 8, ndf * 8, self.lstm),
        #         GeneratorBlock(ndf * 8, ndf * 8, self.lstm),
        #         GeneratorBlock(ndf * 8, ndf * 4, self.lstm),
        #         GeneratorBlock(ndf * 4, ndf * 2, self.lstm),
        #         GeneratorBlock(ndf * 2, ndf, self.lstm),
        #     ]
        # )
        self.blocks = nn.ModuleList(
            [
                GeneratorBlock(ndf * 8, ndf * 8, self.lstm),
                GeneratorBlock(ndf * 8, ndf * 8, self.lstm),
                GeneratorBlock(ndf * 8, ndf * 4, self.lstm),
                GeneratorBlock(ndf * 4, ndf * 2, self.lstm),
                GeneratorBlock(ndf * 2, ndf, self.lstm),
            ]
        )

        self.upsample = nn.Upsample(scale_factor=2)

        self.output_layer = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, c):
        out = self.input_layer(x)
        out = out.view(-1, self.ndf * 8, 4, 4)
        for i, block in enumerate(self.blocks):
            out = block(out, c)
            if i != len(self.blocks) - 1:
                out = self.upsample(out)

        return self.output_layer(out)
