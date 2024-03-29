import cv2
import torch
import torch.nn as nn


class BB_loss(nn.Module):
    def __init__(self, alpha_0=0.5, start_epoch=0, epochs=100, patch_size=3, main_loss=nn.L1Loss, update=True):
        super(BB_loss, self).__init__()
        self.patch_size = patch_size
        self.alpha = alpha_0
        self.start_epoch = start_epoch
        self.end_epoch = start_epoch + epochs
        self.interval = (1 - alpha_0) / epochs
        self.main_loss = main_loss(reduction="none")
        self.alpha_update = update

    def step(self, epoch):
        if epoch >= self.start_epoch and epoch < self.end_epoch and self.alpha:
            self.alpha += self.interval
            print("update alpha to {} in bb loss".format(self.alpha))

    def get_G_database(self, tar):
        if len(tar.shape) == 3:
            tar = tar.unsqueeze(0)
        X2 = torch.nn.functional.interpolate(tar, scale_factor=0.5, mode="bicubic", align_corners=True)
        X4 = torch.nn.functional.interpolate(tar, scale_factor=0.25, mode="bicubic", align_corners=True)
        patches = []
        for i in range(1, self.patch_size):
            for j in range(1, self.patch_size):
                if self.patch_size < min(tar.shape[3], tar.shape[2]):
                    patches.append(self.unfold(tar[:, :, i:-(self.patch_size - i), j:-(self.patch_size - j)]))
                if self.patch_size < min(X2.shape[3], X2.shape[2]):
                    patches.append(self.unfold(X2[:, :, i:-(self.patch_size - i), j:-(self.patch_size - j)]))
                if self.patch_size < min(X4.shape[3], X4.shape[2]):
                    patches.append(self.unfold(X4[:, :, i:-(self.patch_size - i), j:-(self.patch_size - j)]))
        patches.append(self.unfold(tar))
        patches.append(self.unfold(X2))
        patches.append(self.unfold(X4))

        return torch.cat(patches, dim=1)

    def unfold(self, inp):
        if len(inp.shape) == 3:
            inp = inp.unsqueeze(0)
        inp = inp.unsqueeze(1)
        rows = torch.split(inp, split_size_or_sections=self.patch_size, dim=3)
        patches = []
        for i in rows:
            patches.extend(torch.split(i, split_size_or_sections=self.patch_size, dim=4))
        return torch.cat(patches, dim=1)

    def patch_loss(self, m1, m2):
        n = m2.shape[1]
        m1_expand = m1.expand((n, *m1.shape)).transpose(0, 1)
        return torch.sum(torch.nn.functional.mse_loss(m1_expand, m2.unsqueeze(2), reduction='none'), dim=(3, 4, 5))

    def forward(self, inp: torch.Tensor, tar: torch.Tensor, mask=None, weight=None):
        size = tar.shape[2:4]
        inp_patch = self.unfold(inp)
        tar_patch = self.unfold(tar)
        g_candidates = self.get_G_database(tar)
        with torch.no_grad():
            m1 = tar_patch.detach()
            m2 = inp_patch.detach()
            mg = g_candidates.detach()
            l1 = self.patch_loss(m1, mg)
            l2 = self.patch_loss(m2, mg)
            # torch.cuda.empty_cache()
            l = self.alpha * l1 + (1 - self.alpha) * l2
            index = torch.argmin(l, dim=1)
            selected = torch.cat([g_candidates[i, index[i]].unsqueeze(0) for i in range(index.shape[0])], dim=0)

        loss = self.main_loss(inp_patch, selected)

        loss = [torch.split(i, split_size_or_sections=1, dim=1) for i in
                torch.split(loss, split_size_or_sections=size[1] // self.patch_size, dim=1)]
        loss = torch.cat([torch.cat([torch.squeeze(j, 1) for j in i], dim=3) for i in loss], dim=2)

        selected = [torch.split(i, split_size_or_sections=1, dim=1) for i in
                     torch.split(selected, split_size_or_sections=size[1] // self.patch_size, dim=1)]
        selected = torch.cat([torch.cat([torch.squeeze(j, 1) for j in i], dim=3) for i in selected], dim=2)

        return loss.mean(), selected
