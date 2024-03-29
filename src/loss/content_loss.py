import  sys

sys.path.append('../')
from model.crnn import CRNN_cnn, CRNN
import torch

class CRNN_loss(torch.nn.Module):
    def __init__(self, weight_path=None, out_channel=32):
        super(CRNN_loss, self).__init__()
        # self.crnn = CRNN(32, 1, 37, 256)
        self.crnn = CRNN_cnn(32,1)
        if weight_path is not None:
            pretrained_dict = torch.load(weight_path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.crnn.state_dict()}
            self.crnn.load_state_dict(pretrained_dict)
            self.crnn.eval()
        else:
            raise RuntimeError("no crnn weights")



    def preprocess(self, x):
        resize = x
        # resize = torch.nn.functional.interpolate(x, (32, 100), mode='bicubic')
        R = resize[:, 0:1, :, :]
        G = resize[:, 1:2, :, :]
        B = resize[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def forward(self, x, y):
        pre_x = self.preprocess(x)
        pre_y = self.preprocess(y)
        t_x = torch.nn.functional.softmax(self.crnn(pre_x))
        t_y = torch.nn.functional.softmax(self.crnn(pre_y))
        # return torch.nn.functional.l1_loss(t_x, t_y)
        # return torch.nn.functional.l1_loss(pre_x, pre_y.detach())
        return torch.nn.functional.mse_loss(self.crnn(pre_x), self.crnn(pre_y).detach())




class CRNN_out(torch.nn.Module):
    def __init__(self, weight_path):
        super(CRNN_out, self).__init__()
        self.crnn = CRNN_cnn(32, 1)
        if weight_path is not None:
            pretrained_dict = torch.load(weight_path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.crnn.state_dict()}
            self.crnn.load_state_dict(pretrained_dict)
            self.crnn.eval()
        else:
            raise RuntimeError("no crnn weights")

        crnn_pretrained_features = self.crnn.cnn
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(3):  # (3):
            self.slice1.add_module(str(x), crnn_pretrained_features[x])
        for x in range(3, 6):  # (3, 7):
            self.slice2.add_module(str(x), crnn_pretrained_features[x])
        for x in range(6, 12):  # (7, 12):
            self.slice3.add_module(str(x), crnn_pretrained_features[x])
        for x in range(12, 18):  # (12, 21):
            self.slice4.add_module(str(x), crnn_pretrained_features[x])
        for x in range(18, 21):  # (21, 30):
            self.slice5.add_module(str(x), crnn_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out



class Content_loss(torch.nn.Module):
    def __init__(self, weight_path):
        super(Content_loss, self).__init__()
        self.crnn = CRNN_out(weight_path)
        self.criterion = torch.nn.MSELoss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.downsample = torch.nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def preprocess(self, x):
        resize = x
        # resize = torch.nn.functional.interpolate(x, (32, 100), mode='bicubic')
        R = resize[:, 0:1, :, :]
        G = resize[:, 1:2, :, :]
        B = resize[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def forward(self, x, y):
        while x.size()[3] > 4096:
            x, y = self.downsample(x), self.downsample(y)
        x_crnn, y_crnn = self.crnn(self.preprocess(x)), self.crnn(self.preprocess(y))
        loss = 0.0
        for iter, (x_fea, y_fea) in enumerate(zip(x_crnn, y_crnn)):
            loss += self.criterion(x_fea, y_fea.detach())
        return loss
    

class Targeted_Content_loss(torch.nn.Module):
    def __init__(self, weight_path):
        super(Targeted_Content_loss, self).__init__()
        self.crnn = CRNN_out(weight_path)
        self.criterion = torch.nn.MSELoss(reduction="none")
        self.back_weights = [0, 0, 0, 1.0, 1.0]
        self.fore_weights = [1.0, 1.0, 1.0, 1.0,1.0]
        self.downsample = torch.nn.AvgPool2d(2, stride=2, count_include_pad=False)
        self.weights = [1.4, 1.4, 1.4, 0.4, 0.4]

    def preprocess(self, x):
        resize = x
        # resize = torch.nn.functional.interpolate(x, (32, 100), mode='bicubic')
        R = resize[:, 0:1, :, :]
        G = resize[:, 1:2, :, :]
        B = resize[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def forward(self, x, y, mask=None, ema_mask=None):
        while x.size()[3] > 4096:
            x, y = self.downsample(x), self.downsample(y)
        x_crnn, y_crnn = self.crnn(self.preprocess(x)), self.crnn(self.preprocess(y))
        crnn_f_sizes=[i.shape[-2:] for i in x_crnn]
        if mask is not None:
            mask = (mask - torch.min(mask))/(torch.max(mask)-torch.min(mask))
            masks = [torch.nn.functional.interpolate(mask,size=crnn_f_sizes[i])[:,0,:,:] for i in range(len(crnn_f_sizes))]
            masks = [masks[i].unsqueeze(1).expand_as(x_crnn[i]) for i in range(len(x_crnn))]
        
        if ema_mask is not None:
            ema_masks = [torch.nn.functional.interpolate(ema_mask,size=crnn_f_sizes[i])[:,0,:,:] for i in range(len(crnn_f_sizes))]
            ema_masks = [ema_masks[i].unsqueeze(1).expand_as(x_crnn[i]) for i in range(len(x_crnn))]


        loss = 0.0
        for iter, (x_fea, y_fea) in enumerate(zip(x_crnn, y_crnn)):
            if ema_mask is None:
                if mask is not None:
                    loss += ((self.criterion(x_fea, y_fea.detach())*(1-masks[iter]))* self.back_weights[iter] + (self.criterion(x_fea, y_fea.detach())*(masks[iter]))* self.fore_weights[iter]).mean()
                else:
                    loss += self.criterion(x_fea, y_fea.detach()).mean()*self.weights[iter]
            else:
                if mask is not None:
                    loss += (((self.criterion(x_fea, y_fea.detach())*(1-masks[iter]))* self.back_weights[iter] + (self.criterion(x_fea, y_fea.detach())*(masks[iter]))* self.fore_weights[iter]) * ema_masks[iter]).mean()
                else:
                    loss += (self.criterion(x_fea, y_fea.detach()) * ema_masks[iter]).mean()*self.weights[iter]
        return loss