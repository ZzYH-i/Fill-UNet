import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class LRFSA(nn.Module):

    def __init__(self, channels, reduction=2, min_channels=8):
        super(LRFSA, self).__init__()
        inter_channels = max(channels // reduction, min_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.phi_avg = nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False)
        self.phi_max = nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(inter_channels, channels, kernel_size=1, bias=False)

        self.norm1 = LayerNorm2d(inter_channels)
        self.norm2 = LayerNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_feat = self.phi_avg(self.avg_pool(x))   # [B, C', 1, 1]
        max_feat = self.phi_max(self.max_pool(x))   # [B, C', 1, 1]

        avg_feat = self.norm1(avg_feat)
        max_feat = self.norm1(max_feat)

        attn = torch.softmax((avg_feat + max_feat).flatten(1), dim=1).view_as(avg_feat)

        qz = self.proj(attn)                        # [B, C, 1, 1]
        qz = self.norm2(qz)
        qz = self.sigmoid(qz)

        out = x * qz
        return out, qz


class HRFSA(nn.Module):

    def __init__(self, channels, reduction=2, min_channels=8, kernel_size=7):
        super(HRFSA, self).__init__()
        inter_channels = max(channels // reduction, min_channels)

        self.reduce = nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False)
        self.norm1 = LayerNorm2d(inter_channels)

        self.dwconv = nn.Conv2d(
            inter_channels,
            inter_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inter_channels,
            bias=False
        )
        self.pwconv = nn.Conv2d(inter_channels, inter_channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.reduce(x)               # [B, C', H, W]
        feat = self.norm1(feat)
        feat = self.dwconv(feat)
        feat = self.pwconv(feat)

        b, c, h, w = feat.size()

        feat = torch.softmax(feat.view(b, c, -1), dim=-1).view(b, c, h, w)

        qx = self.proj(feat)                # [B, 1, H, W]
        qx = self.sigmoid(qx)

        out = x * qx
        return out, qx


class SCFAM(nn.Module):

    def __init__(self, channels, reduction=2, min_channels=8, use_residual=True):
        super(SCFAM, self).__init__()
        self.lrfsa = LRFSA(channels, reduction=reduction, min_channels=min_channels)
        self.hrfsa = HRFSA(channels, reduction=reduction, min_channels=min_channels)
        self.use_residual = use_residual

    def forward(self, x):
        identity = x

        x, qz = self.lrfsa(x)
        x, qx = self.hrfsa(x)

        if self.use_residual:
            x = x + identity

        return x