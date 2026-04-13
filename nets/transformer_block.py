import math
import torch
import torch.nn as nn


def build_2d_sincos_position_embedding(h, w, dim, device):

    if dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sin-cos position embedding.")

    grid_h = torch.linspace(0, 1, steps=h, device=device)
    grid_w = torch.linspace(0, 1, steps=w, device=device)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')

    pos_dim = dim // 4
    omega = torch.arange(pos_dim, device=device).float() / pos_dim
    omega = 1.0 / (10000 ** omega)

    out_h = grid_h.reshape(-1, 1) * omega.reshape(1, -1)
    out_w = grid_w.reshape(-1, 1) * omega.reshape(1, -1)

    pos_emb = torch.cat([
        torch.sin(out_h), torch.cos(out_h),
        torch.sin(out_w), torch.cos(out_w)
    ], dim=1)

    return pos_emb.unsqueeze(0)   # [1, H*W, dim]


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super(FeedForward, self).__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerEncoderLayer2D(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super(TransformerEncoderLayer2D, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        # x: [B, N, C]
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class BottleneckTransformer(nn.Module):

    def __init__(self, in_channels, embed_dim=512, num_heads=8, num_layers=1, mlp_ratio=4.0, drop=0.0):
        super(BottleneckTransformer, self).__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False) \
            if in_channels != embed_dim else nn.Identity()

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer2D(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop
            )
            for _ in range(num_layers)
        ])

        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False) \
            if in_channels != embed_dim else nn.Identity()

        self.pos_drop = nn.Dropout(drop)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape

        x = self.proj_in(x)              # [B, embed_dim, H, W]
        _, c_new, h, w = x.shape

        x = x.flatten(2).transpose(1, 2).contiguous()   # [B, H*W, C]

        pos_embed = build_2d_sincos_position_embedding(h, w, c_new, x.device)
        x = self.pos_drop(x + pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = x.transpose(1, 2).contiguous().view(b, c_new, h, w)   # [B, C, H, W]
        x = self.proj_out(x)

        x = x + identity
        return x