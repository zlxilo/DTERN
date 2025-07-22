import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
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
