import numpy as np
import torch
from torch import nn
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class MultiHeadPosAttn(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, norm_layer, n_head=4):
        super(MultiHeadPosAttn, self).__init__()
        self.chanel_in = in_dim
        self.dim_feedforward = in_dim * 4
        self.n_head = n_head

        self.query_conv = nn.ModuleList([Conv2d(in_channels=in_dim, out_channels=in_dim//n_head, kernel_size=1) for i in range(n_head)])
        self.key_conv = nn.ModuleList([Conv2d(in_channels=in_dim, out_channels=in_dim//n_head, kernel_size=1) for i in range(n_head)])
        self.value_conv = nn.ModuleList([Conv2d(in_channels=in_dim, out_channels=in_dim//n_head, kernel_size=1) for i in range(n_head)])
        self.feed_forward = nn.Sequential(Conv2d(in_channels=in_dim, out_channels=self.dim_feedforward, kernel_size=1),
                                          nn.PReLU(),
                                          Conv2d(in_channels=self.dim_feedforward, out_channels=in_dim, kernel_size=1),)
        self.norm = norm_layer(in_dim)

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_querys = [self.query_conv[i](x).view(m_batchsize, -1, width*height).permute(0, 2, 1) for i in range(self.n_head)]
        proj_keys = [self.key_conv[i](x).view(m_batchsize, -1, width*height) for i in range(self.n_head)]
        energy = [torch.bmm(proj_querys[i], proj_keys[i]) for i in range(self.n_head)]
        attention = [self.softmax(energy[i]) for i in range(self.n_head)]
        proj_value = [self.value_conv[i](x).view(m_batchsize, -1, width*height) for i in range(self.n_head)]
        
        multihead_out = [torch.bmm(proj_value[i], attention[i].permute(0, 2, 1)) for i in range(self.n_head)]
        multihead_out = [multihead_out[i].view(m_batchsize, -1, height, width) for i in range(self.n_head)]
        multihead_out = torch.cat(multihead_out, 1)
        multihead_out = self.norm(multihead_out + x)
                                                                                              
                                                                                              
        out = self.norm(self.feed_forward(multihead_out) + multihead_out)                                               
                                                                                              
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
    
class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

#         self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
#         self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
#         sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
#         sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

#         output = [sasc_output]
#         output.append(sa_output)
#         output.append(sc_output)
#         return tuple(output)
        return sasc_output