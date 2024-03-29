import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from IPython import embed
from torchvision import transforms
from src.loss.bb_loss import BB_loss
from src.loss.content_loss import CRNN_loss, Content_loss, Targeted_Content_loss
from src.loss.perceptual_loss import VGGLoss

class Mask_loss(torch.nn.Module):
    def __init__(self, main_loss):
        super(Mask_loss, self).__init__()
        self.main_loss = main_loss

    def forward(self, X, Y, mask=None, weight=None, ema_mask=None):
        loss = self.main_loss(X,Y)
        if ema_mask != None:
            loss = loss * ema_mask
        if weight != None:
            loss = loss * weight
        if mask != None:
            loss = loss * mask
            if len(mask.shape) == 4:
                size = (torch.sum(mask) / mask.shape[1]) * loss.shape[1]
            elif len(mask.shape) == 3:
                size = (torch.sum(mask)) * loss.shape[1]
            return torch.sum(loss)/size
        else:
            return torch.mean(torch.mean(torch.mean(loss,dim=-1), dim=-1), dim=-1)


class ImageLoss(nn.Module):
    def __init__(self, gradient=True, crnn_path="", loss_type="mse", loss_weight=[20, 1e-4, 1e-4]):
        super(ImageLoss, self).__init__()
        self.mainLoss = self.main_loss(loss_type)
        self.mainLoss_type = loss_type
        if gradient:
            self.GPLoss = GradientPriorLoss()
        if crnn_path != "":
            self.CRNNLoss = Targeted_Content_loss(crnn_path)
        self.content = (crnn_path != "")

        self.gradient = gradient
        self.loss_weight = loss_weight


    def main_loss(self, loss_type="mse"):
        if loss_type == "mse" or loss_type == "mse_pad":
            return Mask_loss(main_loss=torch.nn.MSELoss(reduction="none"))
        elif loss_type == "l1":
            return Mask_loss(main_loss=torch.nn.L1Loss(reduction="none"))
        elif loss_type == "bb_4":
            return BB_loss(patch_size=4,main_loss=torch.nn.MSELoss)
        elif loss_type == "bb_8":
            return BB_loss(patch_size=8,main_loss=torch.nn.MSELoss)
        else:
            raise RuntimeError("main loss type error")

    def forward(self, out_images, target_images, mask=None, weight=None, boxes=None, texts=None, ema_mask=None):

        metrics = {}
        if self.mainLoss_type == "bb_4" or self.mainLoss_type == "bb_8":
            main_loss, selected = self.mainLoss(out_images, target_images, mask=mask, weight=weight)
        else:
            main_loss = self.mainLoss(out_images, target_images, mask, weight, ema_mask=ema_mask)
            metrics["main_loss"]=main_loss.mean()
            # print(main_loss.mean())

        if self.gradient:
            loss = self.loss_weight[0] *  main_loss+ \
                   self.loss_weight[1] * self.GPLoss(out_images[:, :3, :, :], target_images[:, :3, :, :], ema_mask=ema_mask)
            metrics["gradient loss"] = self.GPLoss(out_images[:, :3, :, :], target_images[:, :3, :, :], ema_mask=ema_mask).mean()
        else:
            loss = self.loss_weight[0] * main_loss

        if self.content:
            loss = loss + self.loss_weight[2] * self.CRNNLoss(out_images[:, :3, :, :], target_images[:, :3, :, :], mask=weight, ema_mask=ema_mask)
            metrics["content loss"] = self.CRNNLoss(out_images[:, :3, :, :], target_images[:, :3, :, :], mask= weight, ema_mask=ema_mask).mean()

        if self.mainLoss_type == "bb_4" or self.mainLoss_type == "bb_8":
            return loss, selected, metrics
        else:
            return loss, metrics


class GradientPriorLoss(nn.Module):
    def __init__(self, ):
        super(GradientPriorLoss, self).__init__()
        self.func = nn.L1Loss(reduction="none")

    def forward(self, out_images, target_images, ema_mask=None):
        map_out = self.gradient_map(out_images)
        map_target = self.gradient_map(target_images)
        if ema_mask is not None:
            return torch.mean(self.func(map_out, map_target) * ema_mask)
        return torch.mean(self.func(map_out, map_target))

    @staticmethod
    def gradient_map(x):
        batch_size, channel, h_x, w_x = x.size()
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2)+1e-6, 0.5)
        return xgrad


if __name__ == '__main__':
    im1=Image.open('../tt.jpg')
    im1=transforms.ToTensor()(im1)
    im1=im1.unsqueeze(0)
    im2 = Image.open('../tt1.jpg')
    im2 = transforms.ToTensor()(im2)
    im2 = im2.unsqueeze(0)
    embed()
