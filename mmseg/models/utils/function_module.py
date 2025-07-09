import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w),mode='bilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), mode='bilinear', align_corners=True)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), mode='bilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), mode='bilinear', align_corners=True)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)
    
class SPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(SPHead, self).__init__()
        inter_channels = in_channels // 2
        self.trans_layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
                norm_layer(inter_channels),
                nn.ReLU(True)
        )
        self.strip_pool1 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.strip_pool2 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
                norm_layer(inter_channels // 2),
                nn.ReLU(True),
                nn.Dropout2d(0.1, False),
                nn.Conv2d(inter_channels // 2, out_channels, 1))

    def forward(self, x):
        x = self.trans_layer(x)
        x = self.strip_pool1(x)
        x = self.strip_pool2(x)
        x = self.score_layer(x)
        return x

class RCA(nn.Module):
    def __init__(self, inp,  kernel_size=1, ratio=1, band_kernel_size=11,dw_size=(1,1), padding=(0,0), stride=1, square_kernel_size=3, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size//2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc=inp//ratio
        self.excite = nn.Sequential(
                nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc),
                nn.BatchNorm2d(gc),
                nn.ReLU(inplace=True),
                nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc),
                nn.Sigmoid()
            )
    
    def sge(self, x):
        #[N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w #.repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather) # [N, 1, C, 1]
        
        return ge

    def forward(self, x):
        loc=self.dwconv_hw(x)
        att=self.sge(x)
        out = att*loc
        
        return out
    


# 构建RF=3、7、11、15的模块 (5,11,17,23) 
class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim) #RF=3
        self.conv0_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, dilation=2 ,groups=dim) #RF=11
        self.conv1 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim) #RF=7
        self.conv1_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, dilation=2 ,groups=dim) #RF=15

        self.conv11 = nn.Conv2d(dim, dim//2, 1)
        self.conv12 = nn.Conv2d(dim, dim//2, 1)
        self.conv21 = nn.Conv2d(dim, dim//2, 1)
        self.conv22 = nn.Conv2d(dim, dim//2, 1)

        self.conv_squeeze = nn.Conv2d(2, 4, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)
 
    def forward(self, x):   
        attn11 = self.conv0(x)              # RF=3
        attn12 = self.conv0_spatial(attn11)   # RF=11
        attn21 = self.conv1(x) # RF=7
        attn22 = self.conv1_spatial(attn21) # RF=15

        attn11 = self.conv11(attn11)
        attn12 = self.conv12(attn12)
        attn21 = self.conv21(attn21)
        attn22 = self.conv22(attn22)
        
        attn = torch.cat([attn11, attn12, attn21, attn22], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn11 * sig[:,0,:,:].unsqueeze(1) + attn21 * sig[:,1,:,:].unsqueeze(1) + attn12 * sig[:,2,:,:].unsqueeze(1) + attn22 * sig[:,3,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
    

# large_k = LSKblock(64)
# input = torch.randn(1, 64, 128, 128)
# output = large_k(input)
# print(output.size())

# SPNet + RCA +PPM +SK

class SRPS(nn.Module):
    def __init__(self, in_channels, pool_size, norm_layer, band_kernel_size=11):
        super(SRPS, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        
        self.excite = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=inter_channels),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=inter_channels),
            )
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(inter_channels, in_channels, 1)

        
        
    def forward(self,x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w),mode='bilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), mode='bilinear', align_corners=True)

        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), mode='bilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), mode='bilinear', align_corners=True)
        # x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        # x2 = self.excite(F.relu_(x2_5 + x2_4))        
        x1 = self.conv2_5(x2_1 + x2_2 + x2_3) # local
        x2 = self.excite(x2_5 + x2_4) # global

        attn = torch.cat([x1, x2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()

        attn = x1 * sig[:,0,:,:].unsqueeze(1) + x2 * sig[:,1,:,:].unsqueeze(1) #selective attention
        attn = self.conv(attn)
        return x * attn




class CAM(nn.Module):
    def __init__(self, in_channels, pool_size, norm_layer, band_kernel_size=11,dw_kernel_size=[11,9,7,5],
                 no_local=False,no_global=False):
        super(CAM, self).__init__()
        # global
        self.no_local = no_local
        self.no_global = no_global
        inter_channels = in_channels
        self.pool3_1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool3_2 = nn.AdaptiveAvgPool2d((None, 1))
        self.conv3_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv3_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.excite = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=inter_channels),
                nn.GELU(),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=inter_channels),
            )

        self.dw_kernel_size = dw_kernel_size
 
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
                        nn.GELU())
        
        self.conv1_2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.GELU(),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1,groups=in_channels)
            )
        
        self.conv2_1 = nn.Sequential(
                        nn.Conv2d(inter_channels, inter_channels, dw_kernel_size[0], 1, dw_kernel_size[0]//2,groups=inter_channels),
                        # nn.GELU(),
                        # nn.Conv2d(inter_channels, inter_channels//4, 1),
                        # nn.GELU(),
                    )
        if len(dw_kernel_size)>1:
            self.conv2_2 = nn.Sequential(
                            nn.Conv2d(inter_channels, inter_channels, dw_kernel_size[1], 1, dw_kernel_size[1]//2,groups=inter_channels)
                        )
        if len(dw_kernel_size)>2:
            self.conv2_3 = nn.Sequential(
                            nn.Conv2d(inter_channels, inter_channels, dw_kernel_size[2],  1, dw_kernel_size[2]//2,groups=inter_channels)
                        )
        if len(dw_kernel_size)>3:
            self.conv2_4 =  nn.Sequential(
                            nn.Conv2d(inter_channels, inter_channels, dw_kernel_size[3], 1, dw_kernel_size[3]//2,groups=inter_channels)
                        )  
                    
        self.conv2_5 =  nn.Sequential(
                        nn.Conv2d(inter_channels, inter_channels, 1),
                        nn.BatchNorm2d(inter_channels),
                        nn.GELU()
                    )  
        self.conv = nn.Sequential(
                        nn.Conv2d(inter_channels, inter_channels, 1),
                        nn.BatchNorm2d(inter_channels),
                        nn.GELU()
                    )  

    def forward(self,x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        # local
        local_out = self.conv2_1(x1)
        if len(self.dw_kernel_size)>1:
            local_out += self.conv2_2(x1)
        if len(self.dw_kernel_size)>2:
            local_out += self.conv2_3(x1)
        if len(self.dw_kernel_size)>3:
            local_out += self.conv2_4(x1)
        
        # local_out = torch.cat([x1,x2_1, x2_2, x2_3, x2_4], dim=1)
        local_out = x1+local_out

        local_out = self.conv2_5(local_out) # local
        
        if self.no_global:
            return self.conv(local_out)
        
        # global
        x3_1 = F.interpolate(self.conv3_1(self.pool3_1(x2)), (h, w), mode='bilinear', align_corners=True)
        x3_2 = F.interpolate(self.conv3_2(self.pool3_2(x2)), (h, w), mode='bilinear', align_corners=True)      
        global_out = self.excite(x3_1 + x3_2) # global
        
        if self.no_local:
            return self.conv(global_out)
        
        attn = torch.sigmoid(global_out)
        return  self.conv(local_out * attn + local_out)



class SRPS3(nn.Module):
    def __init__(self, in_channels, pool_size, norm_layer, band_kernel_size=11):
        super(SRPS3, self).__init__()
        # global
        inter_channels = in_channels
        self.pool3_1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool3_2 = nn.AdaptiveAvgPool2d((None, 1))
        self.conv3_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv3_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.excite = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=inter_channels),
                nn.GELU(),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=inter_channels),
            )

 
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
                        nn.GELU())
        
        self.conv1_2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.GELU(),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1,groups=in_channels)
            )
        
        # 可以考虑split后concat
        self.conv2_1 = nn.Sequential(
                        nn.Conv2d(inter_channels, inter_channels, 11, 1, 5,groups=inter_channels),
                        # nn.GELU(),
                        # nn.Conv2d(inter_channels, inter_channels//4, 1),
                        # nn.GELU(),
                    )
        
        self.conv2_2 = nn.Sequential(
                        nn.Conv2d(inter_channels, inter_channels, 9, 1, 4,groups=inter_channels),
                        # nn.GELU(),
                        # nn.Conv2d(inter_channels, inter_channels//4, 1),
                        # nn.GELU(),
                    )

        self.conv2_3 = nn.Sequential(
                        nn.Conv2d(inter_channels, inter_channels, 7, 1, 3,groups=inter_channels),
                        # nn.GELU(),
                        # nn.Conv2d(inter_channels, inter_channels//4, 1),
                        # nn.GELU(),
                    )

        self.conv2_4 =  nn.Sequential(
                        nn.Conv2d(inter_channels, inter_channels, 5, 1, 2,groups=inter_channels),
                        # nn.GELU(),
                        # nn.Conv2d(inter_channels, inter_channels//4, 1),
                        # nn.GELU(),
                    )  
                    
        self.conv2_5 =  nn.Sequential(
                        nn.Conv2d(2*inter_channels, inter_channels, 1),
                        nn.BatchNorm2d(inter_channels),
                        nn.GELU()
                    )  
        self.conv = nn.Sequential(
                        nn.Conv2d(2*inter_channels, inter_channels, 1),
                        nn.BatchNorm2d(inter_channels),
                        nn.GELU()
                    )  

    def forward(self,x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)

        x2_1,x2_2,x2_3,x2_4 = torch.split(x1, x1.size(1)//4, dim=1)
        # local
        x2_1 = self.conv2_1(x2_1)
        x2_2 = self.conv2_2(x2_2)
        x2_3 = self.conv2_3(x2_3)
        x2_4 = self.conv2_4(x2_4)
        
        local_out = torch.cat([x1,x2_1, x2_2, x2_3, x2_4], dim=1)
        # local_out = x1+x2_1+ x2_2+x2_3+ x2_4

        local_out = self.conv2_5(local_out) # local

        # global
        x3_1 = F.interpolate(self.conv3_1(self.pool3_1(x2)), (h, w), mode='bilinear', align_corners=True)
        x3_2 = F.interpolate(self.conv3_2(self.pool3_2(x2)), (h, w), mode='bilinear', align_corners=True)      
        global_out = self.excite(x3_1 + x3_2) # global
        out = torch.cat([local_out, global_out], dim=1)
        out = self.conv(out)

        return out + x

class Dymfilter(nn.Module):
    '''
        针对高中心噪声的设计：通过低通滤波对噪声进行抑制，为了避免过度的平滑，
    '''
    def __init__(self, in_channels):
        pass


    def forward(self,x):
        pass






# large_k = SRPS2(256, (20, 12), nn.BatchNorm2d, 11)
# input = torch.randn(1, 256, 128, 128)
# output = large_k(input)
# print(output.size())
# # 统计参数数量
# total_params = sum(p.numel() for p in large_k.parameters())
# print(f"Number of parameters: {total_params / 1e6:.2f}M") #0.29M

# rep_cnn = nn.Conv2d(256, 256, 3, padding=1)
# # input = torch.randn(1, 256, 128, 128)
# # output = large_k(input)
# # print(output.size())
# # 统计参数数量
# total_params = sum(p.numel() for p in rep_cnn.parameters())
# print(f"Number of parameters: {total_params / 1e6:.2f}M") #0.29M

