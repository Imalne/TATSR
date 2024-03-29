import sys
import torch
from torch import nn
from torch.nn import functional as F

from src.model.non_local.LineModel import GruBlock, TransformerEncoderBlock, TransformerEncoderBlock2, TransformerEncoderBlock3, RowEmbed, \
    DeRowEmbed
from src.model.non_local.GlocalModel import DANetHead, MultiHeadPosAttn, ViTransformerEncoder, SwinTransformerEncoder, \
    PatchEmbed, DePatchEmbed

sys.path.append('./')
sys.path.append('../')
from .recognizer.tps_spatial_transformer import TPSSpatialTransformer
from .recognizer.stn_head import STNHead
import math


class TSRN(nn.Module):
    def __init__(self, scale_factor=2, width=128, height=32, nonlocal_type="gru", conv_num=2, STN=False, srb_nums=5,
                 mask=True, hidden_units=32):
        super(TSRN, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
            # nn.ReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), self.getNonLocalBlock(nonlocal_type, 2 * hidden_units, conv_num))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2 * hidden_units)
                ))

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [32, 64]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def getNonLocalBlock(self, nonLocalType, channels, conv_num=2):
        if nonLocalType == "gru" or nonLocalType == "transformer" or nonLocalType == "transformer2" or nonLocalType == "transformer3":
            return RecurrentResidualBlock(channels, rnn_type=nonLocalType)
        elif nonLocalType == "DAAttn" or nonLocalType == "MHPosAttn" or nonLocalType == "ViT" or nonLocalType == "SwinTrans":
            return ConvAttnResidualBlock(channels, attn_type=nonLocalType, conv_num=conv_num)
        elif nonLocalType == "patch_gru" or nonLocalType == "patch_transformer" or nonLocalType == "patch_tranformer2":
            return PatchRecurrentResidualBlock(channels, rnn_type=nonLocalType)
        elif nonLocalType == "row_gru" or nonLocalType == "row_transformer" or nonLocalType == "row_transformer2":
            return RowRecurrentResidualBlock(channels, nonLocalType)
        else:
            raise RuntimeError("no such non local block")

    def forward(self, x):
        # embed()
        if self.stn and self.training:
            x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}
        for i in range(self.srb_nums + 1):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3))((block['1'] + block[str(self.srb_nums + 2)]))
        output = torch.tanh(block[str(self.srb_nums + 3)])
        return output


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels, rnn_type="gru"):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = self.getRNNUnit(rnn_type, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = self.getRNNUnit(rnn_type, channels)

    def getRNNUnit(self, unit_type, channels):
        if unit_type == "gru":
            return GruBlock(channels, channels)
        elif unit_type == "transformer":
            return TransformerEncoderBlock(channels, channels)
        elif unit_type == "transformer2":
            return TransformerEncoderBlock2(channels, channels)
        elif unit_type == "transformer3":
            return TransformerEncoderBlock3(channels, channels)
        else:
            raise RuntimeError("no such line block")

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)

        return self.gru2(x + residual)

