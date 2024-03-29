import torch
from torch import nn


class PatchEmbed(nn.Module):
    """ Feature to Patch Embedding
    input : N C H W
    output: N num_patch P^2*C
    """
    def __init__(self, patch_size=1, in_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.dim = self.patch_size ** 2 * in_channels

    def forward(self, x, keep_axis=False):
        N, C, H, W = ori_shape = x.shape

        p = self.patch_size
        if not keep_axis:
            num_patches = (H // p) * (W // p)
            out = torch.zeros((N, num_patches, self.dim)).to(x.device)
            i, j = 0, 0
            for k in range(num_patches):
                if i + p > H:
                    i = 0
                    j += p
                out[:, k, :] = x[:, :, i:i + p, j:j + p].flatten(1)
                i += p
        else:
            out = torch.zeros((N,self.dim, H//p, W//p)).to(x.device)
            for i in range(self.patch_size):
                for j in range(self.patch_size):
                    out[:,(i*self.patch_size+j)*C:(i*self.patch_size+j)*C+C,:,:] = x[:,:,(0+i)::self.patch_size,(0+j)::self.patch_size]

        return out, ori_shape


class DePatchEmbed(nn.Module):
    """ Patch Embedding to Feature
    input : N num_patch P^2*C
    output: N C H W
    """
    def __init__(self, patch_size=1, in_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = None
        self.dim = self.patch_size ** 2 * in_channels

    def forward(self, x, ori_shape, keep_axis=False):
        if not keep_axis:
            N, num_patches, dim = x.shape
            _, C, H, W = ori_shape
            p = self.patch_size
            out = torch.zeros(ori_shape).to(x.device)
            i, j = 0, 0
            for k in range(num_patches):
                if i + p > H:
                    i = 0
                    j += p
                out[:, :, i:i + p, j:j + p] = x[:, k, :].reshape(N, C, p, p)
                # out[:, k, :] = x[:, :, i:i+p, j:j+p].flatten(1)
                i += p
            return out
        else:
            N, Cp, Hp, Wp = x.shape
            B, C, H, W = ori_shape
            assert N == B and Cp == C * (self.patch_size**2) and Hp == H //self.patch_size
            out = torch.zeros(ori_shape).to(x.device)
            for i in range(self.patch_size):
                for j in range(self.patch_size):
                    out[:,:,(0+i)::self.patch_size, (0+j)::self.patch_size] = x[:,(i*self.patch_size+j)*C:(i*self.patch_size+j)*C + C,:,:]
            return out


class Ffn(nn.Module):
    # feed forward network layer after attention
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        N, L, D = q.shape
        q, k, v = self.query(q), self.key(k), self.value(v)
        q = q.reshape(N, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(N, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(N, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, L, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, dim, num_heads, ffn_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.ffn = Ffn(in_features=dim, hidden_features=ffn_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, pos):
        x = self.norm1(x)
        q, k, v = x + pos, x + pos, x
        x = x + self.attn(q, k, v)
        x = x + self.ffn(self.norm2(x))
        return x


if __name__ == '__main__':
    # inp = torch.rand((10,64,16,64))
    # emd = PatchEmbed(4,64)
    # deemd = DePatchEmbed(4,64)
    # e,s = emd(inp, keep_axis=True)
    # out = deemd(e,s, keep_axis=True)
    # print(inp == out)
    pass