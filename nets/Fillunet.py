import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16

from nets.scfam import SCFAM
from nets.transformer_block import BottleneckTransformer
from nets.msfa import MSFA


class unetUp(nn.Module):
    def __init__(self, low_size, high_size, out_size, use_msfa=True, use_scfam=True):
        super(unetUp, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.use_msfa = use_msfa
        self.use_scfam = use_scfam

        if use_msfa:
            self.msfa = MSFA(
                low_channels=low_size,
                high_channels=high_size,
                out_channels=out_size,
                reduction=4
            )
            scfam_channels = out_size
        else:
            self.fuse_conv1 = nn.Conv2d(low_size + high_size, out_size, kernel_size=3, padding=1)
            self.fuse_conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            scfam_channels = out_size

        if use_scfam:
            self.scfam = SCFAM(scfam_channels, reduction=2, use_residual=True)
        else:
            self.scfam = nn.Identity()

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs1, inputs2):

        up_inputs2 = self.up(inputs2)

        if self.use_msfa:
            outputs = self.msfa(inputs1, up_inputs2)
        else:
            outputs = torch.cat([inputs1, up_inputs2], 1)
            outputs = self.fuse_conv1(outputs)
            outputs = self.relu(outputs)
            outputs = self.fuse_conv2(outputs)
            outputs = self.relu(outputs)

        outputs = self.scfam(outputs)
        outputs = self.out_conv(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(
        self,
        num_classes=21,
        pretrained=False,
        backbone='vgg',
        use_transformer=True,
        use_msfa=True,
        use_scfam=True
    ):
        super(Unet, self).__init__()
        self.backbone = backbone

        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)

            encoder_filters = [64, 128, 256, 512, 512]
            bottleneck_channels = 512

        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)

            encoder_filters = [64, 256, 512, 1024, 2048]
            bottleneck_channels = 2048

        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))

        out_filters = [64, 128, 256, 512]

        if use_transformer:
            self.bottleneck_transformer = BottleneckTransformer(
                in_channels=bottleneck_channels,
                embed_dim=512,
                num_heads=8,
                num_layers=1,
                mlp_ratio=4.0,
                drop=0.1
            )
        else:
            self.bottleneck_transformer = nn.Identity()

        self.up_concat4 = unetUp(
            low_size=encoder_filters[3],
            high_size=encoder_filters[4],
            out_size=out_filters[3],
            use_msfa=use_msfa,
            use_scfam=use_scfam
        )
        self.up_concat3 = unetUp(
            low_size=encoder_filters[2],
            high_size=out_filters[3],
            out_size=out_filters[2],
            use_msfa=use_msfa,
            use_scfam=use_scfam
        )
        self.up_concat2 = unetUp(
            low_size=encoder_filters[1],
            high_size=out_filters[2],
            out_size=out_filters[1],
            use_msfa=use_msfa,
            use_scfam=use_scfam
        )
        self.up_concat1 = unetUp(
            low_size=encoder_filters[0],
            high_size=out_filters[1],
            out_size=out_filters[0],
            use_msfa=use_msfa,
            use_scfam=use_scfam
        )

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # bottleneck transformer
        feat5 = self.bottleneck_transformer(feat5)

        # decoder: MSFA + SCFAM
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv is not None:
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
