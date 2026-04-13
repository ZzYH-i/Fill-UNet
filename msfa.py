import torch
import torch.nn as nn
import torch.nn.functional as F


class PWFM(nn.Module):
    def __init__(self, channels, reduction=4):
        super(PWFM, self).__init__()
        hidden = max(channels // reduction, 16)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.local_channel = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.ELU(inplace=True)
        )

        self.weight_gen = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, z):
        # z: [B, C, H, W]
        g = self.global_pool(z)      # [B, C, 1, 1]
        c = self.local_channel(g)    # [B, C, 1, 1]

        # S(Z)
        s = self.weight_gen(g + c)   # [B, C, 1, 1]

        z_refine = z * s + c * g
        return z_refine, s


class MSFA(nn.Module):

    def __init__(self, low_channels, high_channels, out_channels, reduction=4):
        super(MSFA, self).__init__()

        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.high_proj = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.pwfm = PWFM(out_channels, reduction=reduction)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_feat, high_feat):

        i = self.low_proj(low_feat)
        j = self.high_proj(high_feat)

        z = i + j
        z_refine, s = self.pwfm(z)

        k = i * s + (1.0 - s) * j

        out = self.refine(k + z_refine)
        return out