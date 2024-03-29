#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import string
from IPython import embed
import torch.nn.functional as F

def position_enc(inp,d=32):
    b,c,h,w = inp.shape

    p = torch.arange(d).expand(h*w,d).T.view(d,h,w)
    p = (p - p%2)/d*torch.log(torch.tensor(1000).float())

    ph = torch.arange(h).expand(w,h).T.expand(d,h,w)
    px = ph/torch.exp(p)
    px[0::2,:,:] = torch.sin(px[0::2,:,:])
    px[1::2,:,:] = torch.cos(px[1::2,:,:])

    pw = torch.arange(w).expand(h,w).expand(d,h,w)
    py = pw/torch.exp(p)
    py[0::2,:,:] = torch.sin(py[0::2,:,:])
    py[1::2,:,:] = torch.cos(py[1::2,:,:])

    pos_enc = torch.cat((px,py),dim=0).unsqueeze(0).expand(b,2*d,h,w).to(inp.device)

    return torch.cat((inp, pos_enc), axis=1)

def inverse_kernel(k):
    m0, m1, m2, m3, m4, m5 = k.clone().detach().flatten(1).split(1, dim=1)
    det_m = m0 * m4 - m3 * m1
    inverse_k = (torch.cat([m4, -m1, m1 * m5 - m2 * m4, -m3, m0, m2 * m3 - m0 * m5], dim=1) / det_m).reshape(-1, 2, 3)
    return inverse_k


def inverse_affine(img1, img2, k):
    if img1 is None:
        img1 = torch.zeros_like(img2)
    mask = torch.ones_like(img1)
    inverse_k = inverse_kernel(k)

    grid = F.affine_grid(inverse_k, mask.size())
    mask = F.grid_sample(mask, grid)

    img2 = F.grid_sample(img2, grid)
    return mask*img2 + (1-mask) * img1

def generate_rotation_kernel(angle):
    rotate_center = torch.zeros((angle.shape[0], 2, 1))
    M = torch.zeros((angle.shape[0], 2,3))
    M[:,0,0] = torch.cos(angle)
    M[:,1,1] = torch.cos(angle)
    M[:,0,1] = torch.sin(angle)
    M[:,1,0] = -torch.sin(angle)
    M[:,:,2] = -(torch.bmm(M[:,:,:2], rotate_center)-rotate_center).squeeze()
    return M

class STN(nn.Module):
    def __init__(self, input_size, rotate_only=False):
        super(STN, self).__init__()
        self.input_size = input_size
        self.rotate_only = rotate_only

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(input_size[0], input_size[0]//2, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(input_size[0]//2, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        fsize = 10 * ((((input_size[1]-4)//2)-2)//2) * ((((input_size[2]-4)//2)-2)//2)
        self.fc_loc = nn.Sequential(
            nn.Linear(fsize, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2) if not rotate_only else nn.Linear(32, 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        if not self. rotate_only:
            # self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        else:
            self.fc_loc[2].bias.data.copy_(torch.tensor([0.5 , 0.5], dtype=torch.float))
            pass


    def forward(self, x):
        batch_size = x.shape[0]
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        theta = self.fc_loc(xs)
        if not self.rotate_only:
            theta = theta.view(-1, 2, 3)
        else:
            theta = theta.view(-1,2)
            theta = theta/theta.norm(dim=-1, keepdim=True)
            cos = theta[:,0]
            sin = theta[:,1]
            zeros = torch.zeros_like(sin)
            theta = torch.stack((cos, -sin, zeros, sin, cos, zeros), dim=-1).view(-1,2,3)
            # theta = generate_rotation_kernel(torch.atan())
        mask = torch.ones_like(x)
        grid = F.affine_grid(theta, x.size())
        # x = F.grid_sample(x, grid)
        mask = F.grid_sample(mask, grid)
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        return theta, mask

def str_filt(str_, voc_type):
    alpha_dict = {
        'digit': string.digits,
        'lower': string.digits + string.ascii_lowercase,
        'upper': string.digits + string.ascii_letters,
        'all':   string.digits + string.ascii_letters + string.punctuation
    }
    if voc_type == 'lower':
        str_ = str_.lower()
    for char in str_:
        if char not in alpha_dict[voc_type]:
            str_ = str_.replace(char, '')
    return str_


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            from IPython import embed
            # embed()
            text = [
                self.dict[char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    # v.data.resize_(data.size()).copy_(data)
    v.resize_(data.size()).copy_(data)

def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img


if __name__=='__main__':
    converter = strLabelConverter(string.digits+string.ascii_lowercase)
    embed()