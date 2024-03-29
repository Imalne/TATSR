from torch import nn
import torch
from src.utils.util import position_enc


class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mask_rate=0):
        super(TransformerEncoderBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        self.transformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(out_channels, 8, out_channels),
                                                        num_layers=3)
        self.mask_rate = mask_rate


    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        # print(x.shape)
        x = x.view(b[0] * b[1], b[2], b[3])
        # print(x.shape)
        x = x.transpose(1, 0)
        # print(x.shape)
        if self.mask_rate > 0:
            s,n,e = x.shape
            t=torch.ones((n,s)).to(x.device)
            t[:,::int(1/self.mask_rate)]=0
            idx = torch.randperm(t.shape[1])
            idx = torch.randperm(t.shape[1])
            t = t[:, idx].view(t.size())
            t = t==1
            x = self.transformerEncoder(x, src_key_padding_mask=t)
        else:
            x = self.transformerEncoder(x)
        # print(x.shape)
        x = x.transpose(1, 0)
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x


class TransformerEncoderBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransformerEncoderBlock3, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        self.transformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(out_channels, 8, out_channels, dropout=0.2),
                                                        num_layers=3)


    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        # print(x.shape)
        x = x.view(b[0] * b[1], b[2], b[3])
        # print(x.shape)
        x = x.transpose(1, 0)
        # print(x.shape)
        x = self.transformerEncoder(x)
        # print(x.shape)
        x = x.transpose(1, 0)
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x


class TransformerEncoderBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransformerEncoderBlock2, self).__init__()
        self.pos_enc_dim=32
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(out_channels+self.pos_enc_dim*2, out_channels, kernel_size=1, padding=0)

        self.transformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(out_channels+self.pos_enc_dim*2, 8, out_channels+self.pos_enc_dim*2), num_layers=2)

    def forward(self, x):

        x = position_enc(x, self.pos_enc_dim)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        # print(x.shape)
        x = x.view(b[0] * b[1], b[2], b[3])
        # print(x.shape)
        x = x.transpose(1, 0)
        # print(x.shape)

        x = self.transformerEncoder(x)
        # print(x.shape)
        x = x.transpose(1, 0)
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        return x

class RowEmbed(nn.Module):
    def __init__(self, row_len, in_channels):
        super().__init__()
        self.patch_size = row_len
        self.dim = row_len * in_channels

    def forward(self, x, keep_axis=False):
        N, C, H, W = ori_shape = x.shape
        assert H == self.patch_size and self.dim == C * H
        out = torch.zeros((N, self.dim, W)).to(x.device)
        for i in range(H):
            out[:,i*C:(i+1)*C,:] = x[:,:,i,:]
        return out if not keep_axis else out.unsqueeze(2), ori_shape


class DeRowEmbed(nn.Module):
    def __init__(self, row_len=1, in_channels=64):
        super().__init__()
        self.patch_size = row_len
        self.dim = row_len * in_channels

    def forward(self, x, ori_shape, keep_axis=False):
        B,C,H,W = ori_shape
        if keep_axis:
            x = x.squeeze(2)
        assert self.dim == C * H
        out = torch.zeros((ori_shape)).to(x.device)
        for i in range(H):
            out[:,:,i,:] = x[:,i*C:(i+1)*C,:,]
        return out

class TransformerEncoderBlock_with_padding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransformerEncoderBlock_with_padding, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        self.transformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(out_channels, 8, out_channels),
                                                        num_layers=3)

    def forward(self, x, padding_mask):
        x = self.conv1(x)
        x = torch.nn.ZeroPad2d((0,1,0,0))(x)
        padding_mask = torch.nn.ZeroPad2d((0,1,0,0))(padding_mask)
        x = x.permute(0, 2, 3, 1).contiguous()
        padding = padding_mask.permute(0,2,3,1).contiguous()
        b = x.size()
        p = padding.size()
        # print(x.shape)


        x = x.view(b[0] * b[1], b[2], b[3])
        padding = padding.view(p[0] * p[1], p[2], p[3])
        # print(x.shape)
        x = x.transpose(1, 0)
        padding = (padding == 1)[:,:,0]
        # print(x.shape)
        x = self.transformerEncoder(x, src_key_padding_mask=padding)
        # print(x.shape)
        x = x.transpose(1, 0)
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)[:,:,:,:-1]
        return x

if __name__ == '__main__':
    emd = RowEmbed(16, 64)
    deemd = DeRowEmbed(16,64)
    inp = torch.rand((10,64,16,64))
    e,s = emd(inp)
    out = deemd(e,s)
    print(out == inp)
