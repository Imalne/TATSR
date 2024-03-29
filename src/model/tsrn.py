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


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta=0.25):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.beta = beta 
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
    
    def dist(self, x, y):
        return torch.sum(x ** 2, dim=1, keepdim=True) + \
                    torch.sum(y**2, dim=1) - 2 * \
                    torch.matmul(x, y.t())
    
    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h*w, c)
        y = y.reshape(b, h*w, c)

        gmx = x.transpose(1, 2) @ x / (h*w)
        gmy = y.transpose(1, 2) @ y / (h*w)
    
        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization. 
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)

        codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss


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


class ConvAttnResidualBlock(nn.Module):
    def __init__(self, channels, conv_num=2, attn_type="DAAttn"):
        super(ConvAttnResidualBlock, self).__init__()
        self.conv_num = conv_num
        self.attn = self.getAttnUnit(attn_type, channels, nn.BatchNorm2d)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.PReLU()
            )
            for _ in range(conv_num)
        ])

    def getAttnUnit(self, attn_type, channels, norm):
        if attn_type == "DAAttn":
            return DANetHead(channels, channels, norm)
        elif attn_type == "MHPosAttn":
            return MultiHeadPosAttn(channels, norm)
        elif attn_type == "ViT":
            return ViTransformerEncoder(channels, (16, 64), patch_size=4, norm_layer=nn.LayerNorm)
        elif attn_type == "SwinTrans":
            return SwinTransformerEncoder(channels, (16//4, 64//4), patch_size=4, norm_layer=nn.LayerNorm)
        else:
            raise RuntimeError("no such global block")

    def forward(self, x):
        residual = x
        for i in range(self.conv_num):
            residual = self.convs[i](residual)
        residual = self.attn(residual)

        return residual


class PatchRecurrentResidualBlock(nn.Module):
    def __init__(self, channels, rnn_type="patch_gru", patch_size=4):
        super(PatchRecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = self.getRNNUnit(rnn_type, channels, patch_size=patch_size)
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = self.getRNNUnit(rnn_type, channels, patch_size=patch_size)
        self.patch_emd = PatchEmbed(patch_size=patch_size, in_channels=channels)
        self.depatch_emd = DePatchEmbed(patch_size=patch_size, in_channels=channels)

    def getRNNUnit(self, unit_type, channels, patch_size):
        if unit_type == "patch_gru":
            return GruBlock(channels*(patch_size**2), channels*(patch_size**2))
        elif unit_type == "patch_transformer":
            return TransformerEncoderBlock(channels*(patch_size**2), channels*(patch_size**2))
        elif unit_type == "patch_transformer2":
            return TransformerEncoderBlock2(channels*(patch_size**2), channels*(patch_size**2))
        else:
            raise RuntimeError("no such patch line block")

    def forward(self, x):
        x_emd, _ = self.patch_emd(x, keep_axis=True)
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        residual, ori_shape = self.patch_emd(residual, keep_axis=True)
        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)
        residual = self.gru2(x_emd + residual)

        return self.depatch_emd(residual, ori_shape, keep_axis=True)


class RowRecurrentResidualBlock(nn.Module):
    def __init__(self, channels, rnn_type="row_gru", row_len=16):
        super(RowRecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = self.getRNNUnit(rnn_type, channels)
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = self.getRNNUnit(rnn_type, channels * row_len)
        self.row_emd = RowEmbed(row_len=16, in_channels=channels)
        self.derow_emd = DeRowEmbed(row_len=16, in_channels=channels)

    def getRNNUnit(self, unit_type, channels):
        if unit_type == "row_gru":
            return GruBlock(channels, channels)
        elif unit_type == "row_transformer":
            return TransformerEncoderBlock(channels, channels)
        elif unit_type == "row_transformer2":
            return TransformerEncoderBlock2(channels, channels)
        else:
            raise RuntimeError("no such row line block")

    def forward(self, x):
        x_emd, _ = self.row_emd(x, keep_axis=True)
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)
        residual, ori_shape = self.row_emd(residual, keep_axis=True)
        # residual = self.non_local(residual)
        residual = self.gru2(x_emd + residual)

        return self.derow_emd(residual, ori_shape, keep_axis=True)
