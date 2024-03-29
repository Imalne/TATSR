#!/usr/bin/python
# encoding: utf-8
import math
import os
import random
import time

import PIL.Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
import bisect
import warnings
from PIL import Image
import numpy as np
import string
from dataset.utils_blindsr import degradation_bsrgan_plus

from utils import labelmaps

sys.path.append('../')
from src.utils import str_filt
# from utils.labelmaps import get_vocabulary, labels2strs
# from IPython import embed
random.seed(0)

scale = 0.90


def rand_crop(im):
    w, h = im.size
    p1 = (random.uniform(0, w*(1-scale)), random.uniform(0, h*(1-scale)))
    p2 = (p1[0] + scale*w, p1[1] + scale*h)
    return im.crop(p1 + p2)


def central_crop(im):
    w, h = im.size
    p1 = (((1-scale)*w/2), (1-scale)*h/2)
    p2 = ((1+scale)*w/2, (1+scale)*h/2)
    return im.crop(p1 + p2)


def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im


class lmdbDataset(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=31, test=True):
        super(lmdbDataset, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples

        self.max_len = max_len
        self.voc_type = voc_type

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)

        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())

        try:
            img = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
        except TypeError:
            img = buf2PIL(txn, b'image-%09d' % index, 'RGB')
        except IOError or len(word) > self.max_len:
            return self[index + 1]

        label_str = str_filt(word, self.voc_type)
        return img, label_str


class lmdbDataset_real(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False, heat_map=False):
        super(lmdbDataset_real, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test
        self.heat_map = heat_map

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        img_mask_key = b'image_mask-%09d' % index  # 128*32
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')
            img_mask = buf2PIL(txn, img_mask_key, 'RGB') if self.heat_map else None
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        if not self.test:
            return img_HR, img_lr, label_str, img_mask
        else:
            return img_HR, img_lr, label_str, None


class lmdbDataset_real_plus_bsr(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False, heat_map=False, bsr_prob=0.8,scale_factor=2):
        super(lmdbDataset_real_plus_bsr, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.hr_nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test
        self.heat_map = heat_map
        self.bsr_prob = bsr_prob
        self.scale_factor = scale_factor
        self.expand_rate = 1/(1-self.bsr_prob) if self.bsr_prob < 1 else 1
        self.nSamples = int(self.hr_nSamples * self.expand_rate)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = index % self.hr_nSamples
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        img_mask_key = b'image_mask-%09d' % index  # 128*32
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            if np.random.rand() > 1 - self.bsr_prob:
                a,b = degradation_bsrgan_plus(np.array(img_HR)/255, sf=self.scale_factor)
                img_lr = Image.fromarray((a*255).astype(np.uint8))
                img_HR = Image.fromarray((b*255).astype(np.uint8))
            else:
                img_lr = buf2PIL(txn, img_lr_key, 'RGB')
            img_mask = buf2PIL(txn, img_mask_key, 'RGB') if self.heat_map else None
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        if not self.test:
            return img_HR, img_lr, label_str, img_mask
        else:
            return img_HR, img_lr, label_str, None

class lmdbDataset_all_bsr(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False, heat_map=False, bsr_prob=1,scale_factor=2):
        super(lmdbDataset_all_bsr, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.hr_nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test
        self.heat_map = heat_map
        self.bsr_prob = bsr_prob
        self.scale_factor = scale_factor
        self.expand_rate = 1
        self.nSamples = int(self.hr_nSamples * self.expand_rate)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = index % self.hr_nSamples
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        img_mask_key = b'image_mask-%09d' % index  # 128*32
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            a,b = degradation_bsrgan_plus(np.array(img_HR)/255, sf=self.scale_factor)
            img_lr = Image.fromarray((a*255).astype(np.uint8))
            img_HR = Image.fromarray((b*255).astype(np.uint8))
            img_mask = buf2PIL(txn, img_mask_key, 'RGB') if self.heat_map else None
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        if not self.test:
            return img_HR, img_lr, label_str, img_mask
        else:
            return img_HR, img_lr, label_str, None

# class npDataset_real(Dataset):
#     def __init__(self, root=None, voc_type='upper', max_len=100, test=False, heat_map=False):
#         super(npDataset_real, self).__init__()
#         self.hrs = np.load(os.path.join(root,"hr.npy"))
#         self.lrs = np.load(os.path.join(root,"lr.npy"))
#         self.strs = np.load(os.path.join(root,"str.npy"))
#         self.regions = np.load(os.path.join(root,"regions.npy")).astype(np.int)
#         assert self.hrs.shape[0] == self.lrs.shape[0] and self.lrs.shape[0] == self.strs.shape[0] and self.strs.shape[0] == self.regions.shape[0]
#         self.nSamples = self.hrs.shape[0]
#         self.voc_type = voc_type
#         self.max_len = max_len
#         self.test = test
#         self.heat_map = heat_map

#     def __len__(self):
#         return self.nSamples

#     def __getitem__(self, index):
#         assert index <= len(self), 'index range error'
#         img_HR = PIL.Image.fromarray(self.hrs[index][...,::-1])
#         img_lr = PIL.Image.fromarray(self.lrs[index][...,::-1])
#         region_num = int(self.regions[index,0,0])
#         if region_num > 0:
#             regions = self.regions[index,1:region_num+1]
#             regions[:,3] = np.min(np.stack((regions[:,3], img_HR.size[1]-regions[:,1]), axis=1), axis=1)
#         else:
#             regions=[]
#         label_str = self.strs[index]
#         img_mask = None
#         if not self.test:
#             return img_HR, img_lr, label_str, img_mask, regions
#         else:
#             return img_HR, img_lr, label_str


def replicate(lr, hr, mask, str_label, rep_num=2):
    replicate_lr = Image.new(lr.mode, (lr.size[0] * rep_num, lr.size[1]))
    replicate_hr = Image.new(hr.mode, (hr.size[0] * rep_num, hr.size[1]))
    if mask is not None:
        replicate_mask = Image.new(mask.mode, (mask.size[0] * rep_num, mask.size[1]))
    else:
        replicate_mask = None
    replicate_str = ""
    for i in range(rep_num):
        replicate_lr.paste(lr,(i * lr.size[0], 0, (i+1) * lr.size[0], lr.size[1]))
        replicate_hr.paste(hr,(i * hr.size[0], 0, (i+1) * hr.size[0], hr.size[1]))
        if mask is not None:
            replicate_mask.paste(mask,(i * mask.size[0], 0, (i+1) * mask.size[0], mask.size[1]))
        replicate_str += str_label
    # print("replicate input")
    return replicate_lr, replicate_hr, replicate_str, replicate_mask



def random_replicate(lr,hr,str, mask, target_rate):
    if lr.size[0]/lr.size[1] < target_rate[0]/target_rate[1] * 0.5:
        if np.random.rand()>0.5:
            return replicate(lr, hr, mask, str, np.random.randint(2,4))
        else:
            return lr, hr, str, mask
    elif lr.size[0]/lr.size[1] < target_rate[0]/target_rate[1] * 1:
        if np.random.rand()>0.5:
            return replicate(lr, hr, mask, str)
        else:
            return lr, hr, str, mask
    else:
        return lr, hr, str, mask



class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)
        return img_tensor


class lmdbDataset_mix(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_mix, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        if self.test:
            try:
                img_HR = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
                img_lr = buf2PIL(txn, b'image_lr-%09d' % index, 'RGB')
            except:
                img_HR = buf2PIL(txn, b'image-%09d' % index, 'RGB')
                img_lr = img_HR

        else:
            img_HR = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
            if random.uniform(0, 1) < 0.5:
                img_lr = buf2PIL(txn, b'image_lr-%09d' % index, 'RGB')
            else:
                img_lr = img_HR

        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate_syn(object):
    def __init__(self, imgH=64, imgW=256, down_sample_scale=4, keep_ratio=False, min_ratio=1, mask=False, heat_map=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        self.heat_map = heat_map

    def __call__(self, batch):
        images, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)

        images_hr = [transform(image) for image in images]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        images_lr = [image.resize((image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale), Image.BICUBIC) for image in images]
        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        return images_hr, images_lr, label_strs


class alignCollate_real(alignCollate_syn):
    def __init__(self, imgH=64, imgW=256, down_sample_scale=4, keep_ratio=False, min_ratio=1, mask=False, heat_map=False, aug=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        self.heat_map = heat_map
        self.aug = aug

    def __call__(self, batch):
        if self.aug:
            batch = [random_replicate(*i, target_rate=(self.imgW,self.imgH)) for i in batch]
        images_HR, images_lr, label_strs, images_mask = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        if self.heat_map:
            images_mask = [transform(image) for image in images_mask]
            images_mask = torch.cat([t.unsqueeze(0) for t in images_mask], 0)
            thred = 0.3
            images_mask = images_mask * thred + (1 - thred)
            # mask_means = torch.mean(images_mask, dim=(1,2,3), keepdim=True)
            # images_mask = images_mask / mask_means
            # print(torch.mean(images_mask))
            return images_HR, images_lr, label_strs, images_mask
        else:
            return images_HR, images_lr, label_strs

class alignCollate_real_regions(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, label_strs, images_mask, regions = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        if self.heat_map:
            images_mask = [transform(image) for image in images_mask]
            images_mask = torch.cat([t.unsqueeze(0) for t in images_mask], 0)
            thred = 0.3
            images_mask = images_mask * thred + (1 - thred)
            # mask_means = torch.mean(images_mask, dim=(1,2,3), keepdim=True)
            # images_mask = images_mask / mask_means
            # print(torch.mean(images_mask))
            return images_HR, images_lr, label_strs, images_mask, regions
        else:
            return images_HR, images_lr, label_strs, regions


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


def interpolate_kernels(h,w,templates, size, padding_size=0):
    kernel_size = int(size)
    padding_size = int(padding_size)
    templates = [cv2.resize(kernel, dsize=(kernel_size, kernel_size)) for kernel in templates]
    templates = [kernel/np.sum(kernel) for kernel in templates]
    templates = [np.pad(kernel, ((padding_size,padding_size),(padding_size,padding_size))) for kernel in templates]
    lu,ld,ru,rd = templates


    uplist = np.linspace(lu.flatten(), ru.flatten(), num=w)
    downlist = np.linspace(ld.flatten(), rd.flatten(), num=w)
    kernels = np.linspace(uplist, downlist, num=h)
    return kernels

def SVConv(image, templates):
    b,c,h,w = image.shape
    kernel_size = int(math.sqrt(templates.shape[-1]))
    kennels = torch.transpose(torch.transpose(templates, 1,-1),-1,-2)

    image_padded = torch.nn.ReflectionPad2d((kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2))(image)
    result_image = torch.zeros_like(image)
    unfold = torch.zeros((b,kernel_size*kernel_size*c,h,w))
    # for i in range(h):
    #     for j in range(w):
    #         kernel = interpolate_kernel(i,j,h,w, templates)
    #         result_image[i][j] = np.sum(image_padded[i:i+kernel_size, j:j+kernel_size]*kernel, axis=(0,1))
    for i in range(kernel_size):
        for j in range(kernel_size):
            # unfold[:,:,(i*kernel_size+j)::kernel_size*kernel_size] = image_padded[i:i+h,j:j+w,:]
            unfold[:,(i*kernel_size+j)*c:(i*kernel_size+j+1)*c,:,:] = image_padded[:,:,i:i+h,j:j+w]
    for i in range(c):
        result_image[:,i,:,:] = torch.sum(unfold[:,i::c,:,:]*kennels, axis=1)
    return result_image

class SyncDataset(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False, heat_map=False):
        super(SyncDataset, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test
        self.heat_map = heat_map
        self.kernels = np.load(os.path.join(root,"kernels.npy"))

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        img_mask_key = b'image_mask-%09d' % index  # 128*32
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')
            img_mask = buf2PIL(txn, img_mask_key, 'RGB') if self.heat_map else None
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        kernels = self.kernels[np.random.choice(self.kernels.shape[0],size=4,replace=False)]
        if not self.test:
            return img_HR, img_lr, label_str, img_mask, kernels
        else:
            return img_HR, img_lr, label_str, None, kernels

class alignCollate_sync(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, label_strs, mask, kernels = zip(*batch)
        kernel_sizes = [np.clip(np.random.normal(0.4, scale=0.075), 0.1, 0.7) * min(self.imgH, self.imgW) for i in range(len(kernels))]
        kernel_sizes = [kernel_size - kernel_size%2 + 1 for kernel_size in kernel_sizes]
        padding_sizes = (np.max(kernel_sizes) - kernel_sizes)//2
        kernels = torch.stack([torch.tensor(interpolate_kernels(self.imgH, self.imgW, kernels[i], kernel_sizes[i], padding_size=padding_sizes[i])) for i in range(len(kernels))], dim=0)



        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        # images_lr = [transform(image) for image in images_lr]
        images_lr_old = [transform2(image) for image in images_lr]
        images_lr_old = torch.cat([t.unsqueeze(0) for t in images_lr_old], 0)

        images_lr = SVConv(images_HR, kernels)
        images_lr = torch.nn.functional.interpolate(images_lr, scale_factor=0.5)

        label_strs = list(label_strs).extend(label_strs)

        if self.heat_map:
            images_mask = [transform(image) for image in images_mask]
            images_mask = torch.cat([t.unsqueeze(0) for t in images_mask], 0)
            thred = 0.4
            images_mask = images_mask * thred + (1 - thred)
            # mask_means = torch.mean(images_mask, dim=(1,2,3), keepdim=True)
            # images_mask = images_mask / mask_means
            # print(torch.mean(images_mask))
            return images_HR, images_lr, label_strs, images_mask
            # return torch.cat((images_HR, images_HR), dim=0), torch.cat((images_lr, images_lr_old), dim=0), label_strs.extend(label_strs), images_mask
        else:
            return images_HR, images_lr, label_strs
            # return torch.cat((images_HR, images_HR), dim=0), torch.cat((images_lr, images_lr_old), dim=0),label_strs


if __name__ == '__main__':
    dataset = SyncDataset(root="/data/qinrui/Datasets/TextZoom/train1")
    dataloader =DataLoader(dataset,batch_size=4,shuffle=True,collate_fn=alignCollate_sync(32,128,2, mask=False))
    start_time = time.time()
    now = start_time
    for i, data in enumerate(dataloader):
        # continue
        # print(now-start_time)
        # start_time = now
        # now = time.time
        gt, inp, _ = data
       # cv2.imshow("hr", np.transpose(gt[0].numpy(),(1,2,0)))
       # cv2.imshow("lr", np.transpose(inp[0].numpy(),(1,2,0)))
       # cv2.imshow("lr2", np.transpose(inp[4].numpy(),(1,2,0)))

        cv2.imshow("a",np.vstack([
            np.transpose(gt[0].numpy(), (1, 2, 0)),
            cv2.resize(np.transpose(inp[0].numpy(), (1, 2, 0)), dsize=None, fx=2, fy=2),
            cv2.resize(np.transpose(inp[4].numpy(), (1, 2, 0)), dsize=None, fx=2, fy=2)]
        ))
        cv2.waitKey(0)

        # hr,lr,text,_,_ = dataset.__getitem__(i)
        pass
        # hr.show()
        # lr.show()
        # break
