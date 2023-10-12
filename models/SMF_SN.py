import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
import torchsummary


class TAFA(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, mode='embedded_gaussian',
                 sub_sample_factor = 4, bn_layer=True):

        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation', 'concat_proper',
                        'concat_proper_down']

        self.mode = mode
        self.dimension = dimension
        self.sub_sample_factor = sub_sample_factor if isinstance(sub_sample_factor, list) else [sub_sample_factor]

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None

        return output


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU())
        self.aspp_block_1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.aspp_block_2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.aspp_block_3 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, dilation=1)
        self.aspp_block_4 = nn.Conv2d(in_channels, out_channels, 3, 1, 2, dilation=2)
        self.aspp_block_5 = nn.Conv2d(in_channels, out_channels, 3, 1, 3, dilation=3)
        self.net = nn.Sequential(nn.BatchNorm2d(5 * out_channels), nn.ReLU(), nn.Conv2d(5 * out_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        x = self.bn_relu(x)
        output = self.net(
            torch.cat(
                [
                    nn.Upsample(size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)(self.aspp_block_1(x)),
                    self.aspp_block_2(x),
                    self.aspp_block_3(x),
                    self.aspp_block_4(x),
                    self.aspp_block_5(x),
                ],
                dim=1,
            )
        )
        return output




class SMF_SN(nn.Module):
    def __init__(self, config):
        super().__init__()
        size, in_channels, encoder_channels, decoder_channels, out_channels = (config['size'], config['in_channels'], config['encoder_channels'], config['decoder_channels'], config['out_channels'])
        self.pool_list = nn.ModuleList(
            [
                nn.Sequential(nn.BatchNorm2d(encoder_channels[0]), nn.ReLU(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.BatchNorm2d(encoder_channels[1]), nn.ReLU(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.BatchNorm2d(encoder_channels[2]), nn.ReLU(), nn.MaxPool2d(3, 2)),
                nn.Sequential(nn.BatchNorm2d(encoder_channels[3]), nn.ReLU(), nn.MaxPool2d(2, 2)),
            ]
        )
        self.deconv_list = nn.ModuleList(
            [
                nn.Sequential(nn.BatchNorm2d(decoder_channels[0]), nn.ReLU(), nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], 2, 2, 0)),
                nn.Sequential(nn.BatchNorm2d(decoder_channels[1]), nn.ReLU(), nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], 3, 2, 0)),
                nn.Sequential(nn.BatchNorm2d(decoder_channels[2]), nn.ReLU(), nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], 2, 2, 0)),
                nn.Sequential(nn.BatchNorm2d(decoder_channels[3]), nn.ReLU(), nn.ConvTranspose2d(decoder_channels[3], decoder_channels[4], 2, 2, 0)),
            ]
        )
        self.conv_list = nn.ModuleList(
            [
                nn.Sequential(nn.BatchNorm2d(decoder_channels[4]), nn.ReLU(), nn.Conv2d(decoder_channels[4], decoder_channels[3], 2, 2, 0)),
                nn.Sequential(nn.BatchNorm2d(decoder_channels[3]), nn.ReLU(), nn.Conv2d(decoder_channels[3], decoder_channels[2], 2, 2, 0)),
                nn.Sequential(nn.BatchNorm2d(decoder_channels[2]), nn.ReLU(), nn.Conv2d(decoder_channels[2], decoder_channels[1], 3, 2, 0)),
                nn.Sequential(nn.BatchNorm2d(decoder_channels[1]), nn.ReLU(), nn.Conv2d(decoder_channels[1], decoder_channels[0], 2, 2, 0)),
            ]
        )
        self.stem = nn.Sequential(nn.Conv2d(in_channels, encoder_channels[0], 7, 1, 3), nn.BatchNorm2d(encoder_channels[0]), nn.ReLU())
        self.TAFA_block_1 = TAFA(encoder_channels[0], encoder_channels[0], False)
        self.TAFA_block_2 = TAFA(encoder_channels[0], encoder_channels[1], False)
        assert encoder_channels[4] == decoder_channels[0], 'encoder_channels[4] != decoder_channels[0].'
        self.TAFA_block_5 = TAFA(encoder_channels[3], encoder_channels[4], False)
        self.aspp_1 = ASPP(encoder_channels[4], decoder_channels[0])
        self.TAFA_3 = TAFA(encoder_channels[3], decoder_channels[1])
        self.TAFA_4 = TAFA(encoder_channels[2], decoder_channels[2])
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(decoder_channels[0] * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(decoder_channels[0] * 4, decoder_channels[0]),
            nn.ReLU(),
            nn.Linear(decoder_channels[0], decoder_channels[-1]),
            nn.ReLU(),
            nn.Linear(decoder_channels[-1], out_channels),
        )

    def forward(self, x):
        feature_1 = self.TAFA_block_1(self.stem(x))
        feature_2 = self.TAFA_block_2(self.pool_list[0](feature_1))
        feature_3 = self.TAFA_block_3(self.pool_list[1](feature_2))
        feature_4 = self.TAFA_block_4(self.pool_list[2](feature_3))
        feature_5 = self.TAFA_block_5(self.pool_list[3](feature_4))
        feature_5 = self.aspp_1(feature_5)
        output = self.output_layer(torch.cat([feature_5, feature_1, self.TAFA_5(feature_2, feature_4), self.mgca_5(feature_2, feature_3)], dim=1))
        return output


if __name__ == '__main__':
    config = {
        'size': 512,
        'in_channels': 3,
        'encoder_channels': [32, 64, 128, 256, 512],
        'decoder_channels': [512, 256, 128, 64, 32],
        'out_channels': 3,
    }
    net = SMF_SN(config)
    print(net(torch.randn(8, 3, 512, 512)).shape)
    print(torchsummary.summary(net, (3, 512, 512), batch_size=-1, device='cpu'))
