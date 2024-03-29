import PIL.Image
import cv2
import torchvision.transforms
from torch.nn import Module
import torch
import numpy as np
import cv2
import time
import random

def patch2Img(patches, patch_nums):
    w = patch_nums[1]
    h = patch_nums[0]
    assert patches.shape[0] == patch_nums[0] * patch_nums[1]
    split_patch = [ torch.split(line, split_size_or_sections=1) for line in torch.split(patches, split_size_or_sections=20)]
    combines=[]
    for line  in split_patch:
        combines.append(torch.cat(line, dim=-1))
    return np.transpose(torch.cat(combines, dim=-2).numpy()[0],(1,2,0))


def select_patches(gt_region, gt_patches, inp_patches, candidate_patches, masks, patch_size=8, alpha=0.5):
    loss1_8 = torch.mean(
        torch.nn.functional.mse_loss(gt_patches.unsqueeze(1), candidate_patches.expand((gt_patches.shape[0], *candidate_patches.shape)),
                                     reduction="none"), dim=(2, 3, 4))
    loss2_8 = torch.mean(torch.nn.functional.mse_loss(inp_patches.unsqueeze(1),
                                                      candidate_patches.expand((inp_patches.shape[0], *candidate_patches.shape)),
                                                      reduction="none"), dim=(2, 3, 4))
    loss = alpha * loss1_8 + (1- alpha) *loss2_8
    min_loss, selected = torch.min(loss, dim=1)

    # thred = 3 / 255
    #
    valid_patch_index = torch.argsort(min_loss)[:min_loss.shape[0]//2]

    gt_region = patch_replace(gt_region, torch.tensor(masks)[valid_patch_index],
                              candidate_patches[selected][valid_patch_index], patch_size=patch_size)
    return gt_region

def patch_replace(region, masks, patches, patch_size=8):
    assert len(masks) == patches.shape[0]
    zero = torch.zeros_like(region).float()
    mask = zero.clone()
    for i in range(len(masks)):
        mask[:,masks[i][0]:masks[i][0]+patch_size, masks[i][1]:masks[i][1]+patch_size] += 1
        zero[:, masks[i][0]:masks[i][0] + patch_size, masks[i][1]:masks[i][1] + patch_size] += patches[i]
    return (zero+region)/(mask+1)



def str_delete(s, index):
    assert  len(s) > index
    if index == 0:
        return s[1:]
    elif index == len(s) or index == -1:
        return s[:-1]
    else:
        return s[:index] + s[index+1:]

def str_find_all(s, char):
    indexs=[]
    for i in range(len(s)):
        if s[i] == char:
            indexs.append(i)
    return indexs

def get_patch(img):
    patches_8=[]
    masks_8 = []
    patches_4=[]
    masks_4 = []
    # patches_2=[]
    # masks_2 = []
    c,h,w = img.shape
    for i in range(h):
        for j in range(w):
            if i <= h-8 and j<=w-8:
                patches_8.append(img[:,i:i+8, j:j+8])
                masks_8.append((i,j))
            if i <= h-4 and j<=w-4:
                patches_4.append(img[:,i:i+4, j:j+4])
                masks_4.append((i, j))
            # if i <= h-2 and j<=w-2:
            #     patches_2.append(img[:,i:i+2, j:j+2])
            #     masks_2.append((i, j))
    return patches_8, patches_4, masks_8, masks_4

def unfold(inp, patch_size):
    if len(inp.shape) == 3:
        inp = inp.unsqueeze(0)
    inp = inp.unsqueeze(1)
    rows = torch.split(inp, split_size_or_sections=patch_size, dim=3)
    patches=[]
    for i in rows:
        patches.extend(torch.split(i, split_size_or_sections=patch_size, dim=4))
    return torch.cat(patches, dim=1)

def patch_loss(m1, m2):
    n = m2.shape[1]
    m1_expand = m1.expand((n, *m1.shape)).transpose(0,1)
    return torch.sum(torch.nn.functional.mse_loss(m1_expand, m2.unsqueeze(2), reduction='none'), dim=(3,4,5))


# def candidate_select(inp,gt, candidates, patch_size, alpha=0.5):
#     # 将predict 和ground truth 切片
#     inp = unfold(inp, patch_size).detach()
#     gt = unfold(gt, patch_size).detach()
#     # 计算predict patch 和 candidate patch 之间的距离
#     l1 = patch_loss(inp, candidates)
#     l2 = patch_loss(gt, candidates)
#     l = alpha * l1 + (1 - alpha) * l2
#     index = torch.argmin(l, dim=1)
#     selected = torch.cat([candidates[i, index[i]].unsqueeze(0) for i in range(index.shape[0])], dim=0)


class SemanticPairLoss(Module):
    def __init__(self, alpha=0.5, main_loss=torch.nn.L1Loss(), p=0.7):
        super(SemanticPairLoss, self).__init__()
        self.alpha = alpha
        self.main_loss = main_loss
        self.p = p

    # semantic pair for single image
    def text_region_pair(self, inp, gt, boxes, text):
        if len(boxes) <=0 or len(text) <=0:
            return gt
        # 获取文字所在区域
        gt_regions = []
        inp_regions = []



        for box, char in zip(boxes, text):
            gt_regions.append(gt[:, box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
            inp_regions.append(inp[:, box[1]:box[1] + box[3], box[0]:box[0] + box[2]])

        #针对每个字符，进行patch筛选
        for j in range(len(text)):
            # 如果当前图片中存在相同字符：
            candidate_regions = []
            c = text[j]
            gt_region = gt_regions[j]
            inp_region = inp_regions[j]
            left_str = str_delete(text, j)
            if left_str.find(c) >=0:
                indexs = str_find_all(text, c)
                indexs.remove(j)
                for index in indexs:
                    candidate_regions.append(gt_regions[index])

        # 根据representation 的距离，选择最接近的一个
        # TO-DO

            # 将全部候选的candidate region进行patch 切片，从candidate 中选择全部8*8，4*4，2*2尺寸的patch
            if len(candidate_regions)  <= 0:
                continue
            patches_8 = []
            masks_8=[]
            patches_4 = []
            masks_4 = []
            # patches_2 = []
            # masks_2 = []
            for can in candidate_regions:
                p8,p4, m8, m4 = get_patch(can)
                patches_8.extend(p8)
                masks_8.extend(m8)
                patches_4.extend(p4)
                masks_4.extend(m4)
                # patches_2.extend(p2)
                # masks_2.extend(m2)


            # 根据约束在候选patch中选择合理的patch
            gt_patch_8,gt_patch_4,gt_mask_8,gt_mask_4 = get_patch(gt_region)
            patches_8.extend(gt_patch_8)
            patches_4.extend(gt_patch_4)
            # patches_2.extend(gt_patch_2)

            inp_patch_8,inp_patch_4,_,_ = get_patch(inp_region)
            if len(inp_patch_8) >0 and len(gt_patch_8) > 0:
                gt_patch_8 = torch.stack(gt_patch_8)
                inp_patch_8 = torch.stack(inp_patch_8)
                patches_8 = torch.stack(patches_8)
                gt_region = select_patches(gt_region, gt_patch_8, inp_patch_8, patches_8, gt_mask_8, patch_size=8,
                                           alpha=self.alpha)

            if len(inp_patch_4) > 0 and len(gt_patch_4) > 0:
                gt_patch_4 = torch.stack(gt_patch_4)
                inp_patch_4 = torch.stack(inp_patch_4)
                patches_4 = torch.stack(patches_4)
                gt_region = select_patches(gt_region, gt_patch_4, inp_patch_4, patches_4, gt_mask_4, patch_size=4, alpha=self.alpha)

            gt_regions[j] = gt_region

        for i in range(len(gt_regions)):
            gt[:, boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]] = gt_regions[i]
        return gt


    def forward(self, inp, tar, boxes,texts):
        b,c,h,w = inp.shape
        if boxes is not  None and texts is not  None:
            for i in range(b):
                if random.random() > self.p:
                    tar[i] = self.text_region_pair(inp[i].detach(), tar[i].detach().clone(), boxes[i], texts[i])

        return self.main_loss(inp, tar)