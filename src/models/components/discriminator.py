import torch
from torch import nn


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(DiscriminatorBlock, self).__init__()

        self.skip = nn.Sequential(
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, 1)
        )
        if downsample:
            self.skip.append(nn.AvgPool2d(2))

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
            if downsample
            else nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.skip(x) + self.gamma * self.conv(x)


class DiscriminatorFeatureExtractor(nn.Module):
    def __init__(self, ndf):
        super(DiscriminatorFeatureExtractor, self).__init__()

        self.models = nn.Sequential(
            nn.Conv2d(3, ndf, 3, 1, 1),
            DiscriminatorBlock(ndf, ndf * 2),
            DiscriminatorBlock(ndf * 2, ndf * 4),
            DiscriminatorBlock(ndf * 4, ndf * 8),
            DiscriminatorBlock(ndf * 8, ndf * 16),
            # DiscriminatorBlock(ndf * 16, ndf * 16),
            # DiscriminatorBlock(ndf * 16, ndf * 16),
        )

    def forward(self, x):
        return self.models(x)


class Discriminator(nn.Module):
    def __init__(self, ndf, context_dim):
        super(Discriminator, self).__init__()

        self.context_dim = context_dim

        self.joint_conv = nn.Sequential(
            nn.Conv2d(
                ndf * 16 + self.context_dim, ndf * 2, 3, 1, 1, bias=False
            ),  # 512 from clip model
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x, c):
        c = c.view(-1, self.context_dim, 1, 1)
        c = c.repeat(1, 1, 4, 4)
        x_with_c = torch.cat([x, c], dim=1)
        return self.joint_conv(x_with_c)
