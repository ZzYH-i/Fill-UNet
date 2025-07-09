import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


import torch.nn.functional as F

from torch.nn import TransformerEncoderLayer, TransformerEncoder


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)
        self.msfa = MSFA(out_size)



    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


    def forward(self, low_feat, high_feat):
        up_feat = self.up(high_feat)
        concat = torch.cat([low_feat, up_feat], dim=1)
        out = self.relu(self.conv1(concat))
        out = self.relu(self.conv2(out))
        return self.msfa(low_feat, out)



class ChannelConv(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


class PWFM(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 4]):
        super(PWFM, self).__init__()
        self.pool_sizes = pool_sizes
        self.channel_conv1 = ChannelConv(in_channels)
        self.channel_conv2 = ChannelConv(in_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        pooled_feats = []
        for size in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(x, output_size=(size, size))
            pooled = pooled.view(b, c, -1)
            pooled_feats.append(pooled)

        local_info = torch.cat(pooled_feats, dim=-1)
        local_info = self.channel_conv1(local_info)
        local_info = F.leaky_relu(local_info, 0.1)
        local_info = self.channel_conv2(local_info)
        local_info = local_info.mean(-1).view(b, c, 1, 1)

        global_info = F.adaptive_avg_pool2d(x, 1)
        fusion = torch.sigmoid(local_info + global_info)
        return x * fusion


class MSFA(nn.Module):
    def __init__(self, in_channels):
        super(MSFA, self).__init__()
        self.pwfm = PWFM(in_channels)

    def forward(self, low_feat, high_feat):
        attention = torch.sigmoid(self.pwfm(low_feat + high_feat))
        return low_feat * attention + high_feat * (1 - attention)


class SCFAM(nn.Module):
    def __init__(self, in_channels):
        super(SCFAM, self).__init__()
        inter_channels = in_channels // 2

        self.conv_q = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_z = nn.Conv2d(in_channels, inter_channels, kernel_size=1)

        self.conv_out_lr = nn.Conv2d(inter_channels, in_channels, kernel_size=1)

        self.conv_out_hr = nn.Conv2d(inter_channels, 1, kernel_size=1)

        self.ln = nn.LayerNorm([in_channels, 1, 1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.size()

        q_lr = self.conv_q(x)
        v_lr = self.conv_v(x)
        z_lr = self.conv_z(x)

        q_lr_pool = F.adaptive_avg_pool2d(q_lr, (1, 1))
        v_lr_pool = F.adaptive_avg_pool2d(v_lr, (1, 1))

        attention_map_lr = torch.softmax(q_lr_pool * v_lr_pool, dim=1)
        attention_lr = self.conv_out_lr(attention_map_lr)

        out_lr = attention_lr * x

        q_hr = self.conv_q(x)
        q_hr_gp = F.adaptive_avg_pool2d(q_hr, (1, 1))
        q_hr_norm = torch.softmax(q_hr_gp, dim=1)
        v_hr = self.conv_v(x)

        attention_map_hr = self.conv_out_hr(q_hr_norm * v_hr)
        attention_hr = self.sigmoid(attention_map_hr)

        out_hr = attention_hr * out_lr

        return out_hr


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.in_channels = in_channels
        self.transformer_layer = TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=1)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        x = self.transformer_encoder(x)           # (B, N, C)
        x = x.permute(0, 2, 1).view(B, C, H, W)   # (B, C, H, W)
        return x



class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg' ):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]


        # upsampling
        64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        self.transformer = TransformerBlock(in_channels=in_filters[-1])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)


        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True