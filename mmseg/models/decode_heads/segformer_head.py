# segformer head
# to do list:
# Clean up the code

import collections
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead, BaseDecodeHead_clips,BaseDecodeHead_clips2
from mmseg.models.utils import *
import attr

from IPython import embed

import cv2
from .dtern import DTERN 

from .utils.utils import save_cluster_labels
import time
from ..builder import build_loss
from torch.nn import functional as F


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2) #(bt,c,h*w) -> (bt,h*w,c)
        x = self.proj(x) #(bt,h*w,c) -> (bt,h*w,embed_dim)
        return x


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='GN', num_groups=1)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        print(c1.shape, c2.shape, c3.shape, c4.shape)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        # print(torch.cuda.memory_allocated(0))

        return x

@HEADS.register_module()
class SegFormerHead_clips2_DTERN_ensemble4(BaseDecodeHead_clips2):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        print("in SegFormerHead_clips2_DTERN_ensemble4")
        super(SegFormerHead_clips2_DTERN_ensemble4, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        cityscape = kwargs['cityscape']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)


        self.cross_method = kwargs['cross_method'] 
        self.num_clusters = kwargs['num_cluster']
        backbone = kwargs['backbone']
        self.num_layer=2
        self.linear_pred1 = nn.Conv2d(embedding_dim*2, self.num_classes, kernel_size=1)

        print("-------in model: cross_method:",self.cross_method) 
        print("---------self.num_class:",self.num_classes)

        self.dtern=DTERN(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
                                                 cross_method=self.cross_method,num_clusters=self.num_clusters,backbone=backbone,
                                                 cityscape=cityscape) # linear_qkv or cnn_qk
        self.self_ensemble2=True

    def forward(self, inputs, batch_size=None, num_clips=None, img_metas=None):
        if self.training:
            assert self.num_clips==num_clips
        x = self._transform_inputs(inputs)  
        c1, c2, c3, c4 = x 
        n, _, h, w = c4.shape

        _c41 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) 
        _c42 = resize(_c41, size=c1.size()[2:],mode='bilinear',align_corners=False) 

        _c31 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c32 = resize(_c31, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c21 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c22 = resize(_c21, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c12 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c42, _c32, _c22, _c12], dim=1)) 

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w) 
        if not self.training and num_clips!=self.num_clips:
            return x[:,-1]

        start_time1=time.time()
        shape_c1, shape_c2, shape_c3, shape_c4=c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:] #(h,w)
        c1=c1.reshape(batch_size, num_clips, -1, c1.shape[-2], c1.shape[-1])
        c2=c2.reshape(batch_size, num_clips, -1, c2.shape[-2], c2.shape[-1])
        c3=c3.reshape(batch_size, num_clips, -1, c3.shape[-2], c3.shape[-1])
        c4=c4.reshape(batch_size, num_clips, -1, c4.shape[-2], c4.shape[-1])
        query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1] 
        supp1,supp2,supp3,supp4=c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:] 
        query_frame=[query_c1, query_c2, query_c3, query_c4]
        supp_frame=[supp1, supp2, supp3, supp4]

        h2=int(h/2)
        w2=int(w/2)
        _c = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False) 
        _c_further=_c.reshape(batch_size, num_clips, -1, h2, w2)
        
        out_cls_mid = None
        supp_feats,out_cls_mid,cluster_centers,mem_out,assigned_results=self.dtern(query_frame, supp_frame,img_metas = img_metas)

        _c_further2=torch.cat([_c_further[:,-1], supp_feats[0]],1)
        x2 = self.dropout(_c_further2)

        
        x2 = self.linear_pred1(x2)
        x2=resize(x2, size=(h,w),mode='bilinear',align_corners=False)
        x2=x2.unsqueeze(1)

        x3 = resize(out_cls_mid, size=(h,w),mode='bilinear',align_corners=False).unsqueeze(1)
        output=torch.cat([x,x2,x3],dim=1)   

        if not self.training:
            return x2.squeeze(1)
        return output,cluster_centers

class small_decoder2(nn.Module):

    def __init__(self,
                 input_dim=256, hidden_dim=256, num_classes=124,dropout_ratio=0.1):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_classes=num_classes

        self.smalldecoder=nn.Sequential(
            # ConvModule(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True)),
            # ConvModule(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True)),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(hidden_dim, self.num_classes, kernel_size=1)
            )
        # self.dropout=
        
    def forward(self, input):

        output=self.smalldecoder(input)

        return output
