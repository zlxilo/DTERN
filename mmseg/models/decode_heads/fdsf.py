from math import log
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# 特征差分选择融合
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

    
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim,out_channels=None, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        out_channels = out_channels or dim
        self.conv = nn.Conv2d(dim, out_channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class SFD(nn.Module):
    def __init__(self,in_channels,out_channels=None):
        super(SFD, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        # eucb
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1,bias=False,groups=in_channels)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, self.in_channels, kernel_size=1)
        self.fusion = CGAFusion(self.in_channels,self.out_channels,reduction=4)

    def forward(self, x,y): #y-high
        y = self.dw(y)
        y = self.relu(y)
        y = channel_shuffle(y, self.in_channels)
        y = self.conv2(y)
        out = self.fusion(x,y)
        return out


class FDSF(nn.Module):
    def __init__(self,in_channels,out_channels=None):
        super(FDSF, self).__init__()
        self.in_channels = in_channels
        out_channels = out_channels or in_channels
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.branch1_f = SFD(in_channels,out_channels)
        self.branch2_f = SFD(in_channels,out_channels)
        self.out_conv = nn.Conv2d(2*out_channels, out_channels, kernel_size=3,padding=1)    
    
    def forward(self, x,y): #y-high semantic feature, x-low semantic feature
        if x.shape != y.shape:
            y = F.interpolate(y, size=x.shape[2:], mode='nearest')
        branch1 = x - self.avgpool(y)
        branch2 = y - self.maxpool(x)
        branch1_out = self.branch1_f(branch1,y)
        branch2_out = self.branch2_f(x,branch2)
        out = torch.concat([branch1_out,branch2_out],dim=1)
        # out = torch.concat([x,branch1_out,branch2_out,y],dim=1)

        out = self.out_conv(out)
        return out
    
# self.pag3 = PagFM(planes * 2, planes)
class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        self.f_y = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        if with_channel:
            self.up = nn.Sequential(
                                    nn.Conv2d(mid_channels, in_channels, 
                                              kernel_size=1, bias=False),
                                    BatchNorm(in_channels)
                                   )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, y): # y- high semantic feature, x-low semantic
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        
        y_q = self.f_y(y)
        if x.shape != y.shape:    
            y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)##上采样到与x分辨率相同
        x_k = self.f_x(x)
        
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1)) 
        ##dim=1:逐通道相加，假设x_k * y_q的shape为[4, 32, 32, 64]，相加后shape变为[4, 32, 64]，再通过unsqueeze(1)升维为[4, 1, 32, 64]
        if x.shape != y.shape:
            y = F.interpolate(y, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)##上采样到与x分辨率相同
        x = (1-sim_map)*x + sim_map*y
        
        return x


class ConcatFM(nn.Module):
    def __init__(self,in_channels,mid_channels=None,out_channels=None):
        super(ConcatFM, self).__init__()
        self.in_channels = in_channels
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=3 * in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        w = torch.sigmoid(self.conv1(torch.cat((x1, x2), dim=1)))
        feat_x1 = torch.mul(x1, w)
        feat_x2 = torch.mul(x2, w)
        x = self.conv3(torch.cat((self.conv2(feat_x1 + feat_x2), x1, x2), dim=1))
        return x


# split wise fusion, then aggregate,分离高频和低频信息
class SWFG(nn.Module):
    def __init__(self,in_channels,mid_channels=None,out_channels=None):
        super(SWFG, self).__init__()
        self.in_channels = in_channels
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels
        t = int(abs((log(in_channels, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.high_levl_feature = nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1)
        self.low_levl_feature = nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, in_channels, 1, bias=False)
                )
        self.sigmoid = nn.Sigmoid()
        # self.fusion_ll = PagFM(in_channels,in_channels) #感觉这里不能PagFM，应该直接concat
        self.fusion_ll = ConcatFM(in_channels,in_channels)
        self.out = nn.Sequential(
                    nn.Conv2d(in_channels*2, in_channels, 1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                )

    def forward(self,x,y): #y-high semantic feature, x-low semantic feature
        if x.shape != y.shape:
            y = F.interpolate(y, size=x.shape[2:], mode='nearest')
        
        high_x = self.high_levl_feature(x)
        low_x = self.low_levl_feature(x)
        gap_x = self.gap(x)
        weights =  self.sigmoid(self.mlp(gap_x))
        x1 = high_x * weights
        detail_x = low_x * (1-weights)
        out = self.fusion_ll(x1,y)
        out = self.out(torch.cat([out,detail_x],dim=1))
        return out



class SWFG2(nn.Module): # 在PSPNET上测试最好
    def __init__(self,in_channels,mid_channels=None,out_channels=None):
        super(SWFG2, self).__init__()
        self.in_channels = in_channels
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.map = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, in_channels, 1, bias=False)
                )
        self.sigmoid = nn.Sigmoid()
        self.fusion_ll = PagFM(in_channels,in_channels)

    def forward(self,x,y): #y-high semantic feature, x-low semantic feature
        if x.shape != y.shape:
            y = F.interpolate(y, size=x.shape[2:], mode='nearest')
        gap_x = self.gap(x)
        map_x = self.map(x)
        weights =  self.sigmoid(self.mlp(gap_x) + self.mlp(map_x))
        x1 = x * weights
        detail_x = x * (1-weights)
        out = self.fusion_ll(x1,y)
        out = out + detail_x 
        return out

# freq split 
class MBB(nn.Module):
    def __init__(self,in_channels,out_channels=None):
        super(MBB, self).__init__()
        # for high
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,1,1,bias=False),
            nn.GELU()) 
        # for low
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,1,1,bias=False),nn.GELU(),nn.Conv2d(in_channels,in_channels,3,1,1,bias=False),nn.GELU(),nn.Conv2d(in_channels,in_channels,3,1,1,bias=False),nn.GELU())
        
        self.alpha=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.ones(1))
    def forward(self,x):
        return self.alpha*self.conv1(x)+self.beta*self.conv2(x)

class HFM(nn.Module):
    def __init__(self,in_channels,out_channels=None):
        super(HFM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.ca = ChannelAttention(in_channels)
        self.sigmoid = nn.Sigmoid()
        self.PConv_x =  nn.Conv2d(in_channels, in_channels, kernel_size=1,bias=False)
        self.PConv_y =  nn.Conv2d(in_channels, in_channels, kernel_size=1,bias=False)
        
        # 一致区域挖掘
        self.cat_conv = nn.Sequential(
            nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=5,padding=2,groups=2*in_channels,bias=False), #获取相关性 
            nn.BatchNorm2d(2*in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*in_channels, in_channels, kernel_size=1,bias=False),
        )
        self.sa = SpatialAttention()
        self.gap = nn.AvgPool2d(kernel_size=2)
        self.x_y_conv = nn.Sequential(
                            nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1,groups=in_channels,bias=False), #获取相关性 
                            nn.BatchNorm2d(in_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels, in_channels, kernel_size=1,bias=False),
                        )
        self.x_y_conv = nn.Sequential(
                            nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1,groups=in_channels,bias=False), #获取相关性 
                            nn.BatchNorm2d(in_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels, in_channels, kernel_size=1,bias=False),
                        )
        self.sub_conv = nn.Sequential(
                            nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1,groups=in_channels,bias=False), #获取相关性 
                            nn.BatchNorm2d(in_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels, in_channels, kernel_size=1,bias=False)
                            )
        
        # self.out = nn.Conv2d(3*in_channels, out_channels, kernel_size=1,bias=False)
        self.out = nn.Sequential(
                            nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3,padding=1,bias=False), #获取相关性 
                            nn.BatchNorm2d(2*in_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(2*in_channels, in_channels, kernel_size=1,bias=False),
                        )
            
            
    def forward(self, x,y): #y-high
        x = self.PConv_x(x)
        y = self.PConv_y(y)        
        cat_x_y = torch.cat([x,y],dim=1)
        cat_x_y = self.cat_conv(cat_x_y)
        cat_x_y = cat_x_y * self.sigmoid(self.sa(cat_x_y)) # 一致区域挖掘
        sub_x_y = x - y
        sub_y_x = y - x
        sub_x_y = sub_x_y*self.sigmoid(self.ca(sub_x_y))
        sub_y_x = sub_y_x*self.sigmoid(self.ca(sub_y_x))
        sub_x_y = self.x_y_conv(sub_x_y)
        sub_y_x = self.x_y_conv(sub_y_x)
        # out = torch.cat([x , sub_x_y , cat_x_y],dim=1)
        sub_out = sub_x_y + sub_y_x
        sub_out = self.sub_conv(sub_out)
        out = self.out(torch.cat([sub_out,cat_x_y],dim=1))
        return out


class LFM(nn.Module):
    def __init__(self,in_channels,out_channels=None):
        super(LFM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.sa = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        # invert 
        self.x_conv = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, 1,bias=False),
            nn.BatchNorm2d(2*in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*in_channels, 2*in_channels, 3, padding=1,bias=False,groups=2*in_channels),
            nn.BatchNorm2d(2*in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*in_channels, in_channels, 1)
        ) 
        self.y_conv = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, 1,bias=False),
            nn.BatchNorm2d(2*in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*in_channels, 2*in_channels, 3, padding=1,bias=False,groups=2*in_channels),
            nn.BatchNorm2d(2*in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*in_channels, in_channels, 1)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.cat_conv = nn.Sequential(
                            nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3,padding=1,bias=False), #获取相关性 
                            nn.BatchNorm2d(2*in_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(2*in_channels, in_channels, kernel_size=1,bias=False),
                        )
    def forward(self, x,y):
        x = self.x_conv(x)
        y = self.y_conv(y)
        out = torch.cat([x,y],dim=1)
        out = self.cat_conv(out)
        out = out * self.sigmoid(self.sa(out)) 
        return out

class FSFM(nn.Module):
    def __init__(self,in_channels,mid_channels=None,out_channels=None):
        super(FSFM, self).__init__()
        self.in_channels = in_channels
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels
        self.gap = nn.AvgPool2d(kernel_size=2)
        self.fusion_ll = LFM(in_channels,in_channels)
        self.fusion_hh = HFM(in_channels,in_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.ca = ChannelAttention(in_channels)
        self.sigmoid = nn.Sigmoid()
        self.cat_conv = nn.Sequential(
                            nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3,padding=1,bias=False), #获取相关性 
                            nn.BatchNorm2d(2*in_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(2*in_channels, in_channels, kernel_size=1,bias=False),
                        )
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False)
    def forward(self,x,y): #y-high semantic feature, x-low semantic feature
        if x.shape != y.shape:
            y = F.interpolate(y, size=x.shape[2:], mode='nearest')
        l_x = self.gap(x)
        h_x = x - self.up(l_x)
        l_y = self.gap(y)
        h_y = y - self.up(l_y)
        out_h = self.fusion_hh(h_x,h_y)
        out_l = self.up(self.fusion_ll(l_x,l_y))
        out = torch.cat([out_l,out_h],dim=1)
        out = self.cat_conv(out)
        out = self.out(out * self.sigmoid(self.ca(out))) + x
        return out



class DU(nn.Module):
    def __init__(self,in_channels,mid_channels=None,out_channels=None):
        super(DU, self).__init__()
        self.in_channels = in_channels
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels
        self.gap = nn.AvgPool2d(kernel_size=2)
        self.x = nn.Conv2d(in_channels, in_channels, kernel_size=1,bias=False)
        self.y = nn.Conv2d(in_channels, in_channels, kernel_size=1,bias=False)

        self.sa = SpatialAttention() # 这里用ca还是sa
        self.ca = ChannelAttention(in_channels)
        self.sigmoid = nn.Sigmoid()
        self.cat_conv = nn.Sequential(
                            nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3,padding=1,bias=False), #获取相关性 
                            nn.BatchNorm2d(2*in_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(2*in_channels, in_channels, kernel_size=1,bias=False),
                            nn.BatchNorm2d(in_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels, in_channels, kernel_size=1,bias=False),
                        )
        self.relu = nn.GELU()
        self.hx = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,1,1)
        )
        self.out1 = nn.Sequential(
                            nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3,padding=1,bias=False), #获取相关性 
                            nn.BatchNorm2d(2*in_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(2*in_channels, in_channels, kernel_size=1,bias=False),
                        )

    def forward(self,x,y): #y-high semantic feature, x-low semantic feature
        l_x = self.gap(x)
        l_x = self.x(l_x)
        l_y = self.y(y)
        if l_x.shape != l_y.shape:
            l_x = F.interpolate(l_x, size=l_y.shape[2:], mode='bilinear',align_corners=False)
        # print(l_x.shape,l_y.shape)
        
        ll = self.cat_conv(torch.cat([l_x,l_y],dim=1))
        ll = ll * self.sigmoid(self.sa(ll))
        if x.shape != l_x.shape:
            l_x = F.interpolate(l_x, size=x.shape[2:], mode='bilinear',align_corners=False)
        h_x = x - l_x
        if ll.shape != h_x.shape:
            ll = F.interpolate(ll, size=h_x.shape[2:], mode='bilinear',align_corners=False)
        h_x = self.hx(h_x)
        out = self.out1(torch.cat([h_x ,ll],dim=1))
        out = out * self.sigmoid(self.ca(out)) + x
        return out
    
class DU2(nn.Module): #测试最有效的结果,split的融合
    def __init__(self,in_channels,mid_channels=None,out_channels=None):
        super(DU2, self).__init__()
        self.in_channels = in_channels
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels
        self.gap = nn.AvgPool2d(kernel_size=2)
        self.x = nn.Conv2d(in_channels, in_channels, kernel_size=1,bias=False)
        self.y = nn.Conv2d(in_channels, in_channels, kernel_size=1,bias=False)

        self.ca1 = ChannelAttention(in_channels) # 这里用ca还是sa
        self.ca2 = ChannelAttention(in_channels)
        self.sigmoid = nn.Sigmoid()
        self.cat_conv = nn.Conv2d(2*in_channels, in_channels, kernel_size=1,bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.GELU()
        self.hx = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,1,1),
            nn.BatchNorm2d(in_channels),
        )
        self.out1 = nn.Conv2d(2*in_channels, in_channels, kernel_size=1,bias=False)

    def forward(self,x,y): #y-high semantic feature, x-low semantic feature
        l_x = self.gap(x)
        l_x = self.x(l_x)
        l_y = self.y(y)
        if l_x.shape != l_y.shape:
            l_x = F.interpolate(l_x, size=l_y.shape[2:], mode='bilinear',align_corners=False)
        # print(l_x.shape,l_y.shape)
        
        ll = self.cat_conv(torch.cat([l_x,l_y],dim=1))
        ll = self.bn(ll)
        ll = ll * self.sigmoid(self.ca1(ll))
        if x.shape != l_x.shape:
            l_x = F.interpolate(l_x, size=x.shape[2:], mode='bilinear',align_corners=False)
        h_x = x - l_x
        if ll.shape != h_x.shape:
            ll = F.interpolate(ll, size=h_x.shape[2:], mode='bilinear',align_corners=False)
            
        h_x = self.hx(h_x)
        out = self.out1(torch.cat([h_x ,ll],dim=1))
        out = out * self.sigmoid(self.ca2(out)) + x
        return out                  




class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out



# from thop import profile
# from thop import clever_format
# x = torch.randn(2, 64, 64, 64)
# y = torch.randn(2, 64, 64, 64)
# model = DU(64)
# # out = model(x, y)
# # print(out.shape)
# # 使用thop分析模型的运算量和参数量
# flops, params = profile(model, inputs=(x, y))

# # 将结果转换为更易于阅读的格式
# flops, params = clever_format([flops, params], '%.3f')

# print(f"运算量：{flops}, 参数量：{params}")
