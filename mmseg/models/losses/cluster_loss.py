# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from ..builder import LOSSES
import torch.nn.functional as F

@LOSSES.register_module()
class ClusterLoss(nn.Module):
    def __init__(self,
                class_num,
                temperature, 
                multi_views=False,
                loss_name='loss_cluster'):
        super(ClusterLoss,self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.margin = 0.2

        # self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self._loss_name = loss_name
        self.multi_views = multi_views
        self.logit_scale = nn.Parameter(torch.ones([])*np.log(1/0.07)) # 标量可学习参数，计算相似度时的系数

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask
    
    def cal_clip_loss(self,image_features, text_features):
        # 传入的是(n,c) n表示样本数目即聚类的类别数目,c表示特征维度
        device = image_features.device
        image_features = image_features / image_features.norm(dim=1, keepdim=True) # L2正则化，就是直线距离(欧式距离)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp() # 按初始值的话，先log再exp，就是得到1/0.07

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t() # (A@B.T).T == A.T@B A乘B的转置的转置 等于 A的转置乘B
        labels = torch.arange(image_features.shape[0]).long().to(device) # 对角线上为对应的正样本，正好是0, 1, 2...
        
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss
    
    def forward_clip(self,clusters):
        B,t,_,_ = clusters.shape
        # print("clusters.shape",clusters.shape)
        total_loss = 0
        for b in range(B):
            # 遍历视图
            for i in range(t):
                for j in range(i+1,t):
                    c_i_b = clusters[b,i]
                    c_j_b = clusters[b,j]
                    loss = self.cal_clip_loss(c_i_b, c_j_b)
                    # 累加每个batch的损失
                    total_loss += loss
        return total_loss / B

    def forward_ContrastiveLossCosine(self, X1, X2):
        """
            X1: 模态1的特征矩阵, 形状为 (N, C)
            X2: 模态2的特征矩阵, 形状为 (N, C)
        """
        # 对特征进行 L2 归一化
        X1_normalized = F.normalize(X1, p=2, dim=1)  # 归一化 X1
        X2_normalized = F.normalize(X2, p=2, dim=1)  # 归一化 X2
        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(X1_normalized.unsqueeze(1), X2.unsqueeze(0), dim=2)  # 计算 X1 和 X2 之间的余弦相似度
        N = X1.size(0)  # N 是类别数
        loss = 0.0
        for i in range(N):
            for j in range(N):
                if i == j:
                    # 同类别之间的损失，拉近
                    loss += (1 - cosine_similarity[i, j])  # 目标是最大化相似度，余弦相似度越大越好
                else:
                    # 不同类别之间的损失，拉远
                    loss += torch.max(torch.tensor(0.0), self.margin - cosine_similarity[i, j]) ** 2  # 拉远不同类别
        return loss / (N * N)

    def forward(self,clusters,alpha=1.0):
        # B,t,_,c = clusters.shape
        # print("clusters.shape",clusters.shape)
        # total_loss = 0
        # for b in range(B):
        #     # 遍历视图
        #     for i in range(t):
        #         for j in range(i+1,t):
        #             c_i_b = clusters[b,i]
        #             c_j_b = clusters[b,j]
        #             loss = self.forward_ContrastiveLossCosine(c_i_b, c_j_b) 
        #             # 累加每个batch的损失
        #             total_loss += loss
        # return total_loss / B
    
        # return self.forward_multi_views(clusters,alpha)
        return self.forward_clip(clusters)



    #预测目标帧率,实际的多视图计算是有多少个视图就两两都交互计算
    def forward_multi_views(self, clusters,alpha=1.0): #clusters :[b,t,n,c] 1,2,3,4 n:样本数目,c类别数目
        B,t,_,c = clusters.shape
        if not self.multi_views:
            if t == 4: 
                c_i = (clusters[:,0]+clusters[:,1]+clusters[:,2])/3.
                c_j = clusters[:,3]
            else:
                c_i = (clusters[:,0]+clusters[:,1])/2.
                c_j = (clusters[:,2]+clusters[:,3])/2.            
            # softmax
            c_i = c_i.softmax(dim=-1)
            c_j = c_j.softmax(dim=-1)
        else:
            clusters = clusters.softmax(dim=-1)
        #输入结果需要经过softmax
        # 输入形状为 (B, N, C)，其中 B 是 batch size,N是点集合,C 是类簇数

        N = 2 * self.class_num
        total_loss = 0
        
        if not self.multi_views:
            for b in range(B):
                c_i_b = c_i[b]  # (N, C)
                c_j_b = c_j[b]  # (N, C)
                loss,ne_loss = self.cal_loss(c_i_b, c_j_b)
                # 累加每个batch的损失
                total_loss += loss + alpha * ne_loss

        else:
            for b in range(B):
                # 遍历视图
                for i in range(t):
                    for j in range(i+1,t):
                        c_i_b = clusters[b,i]
                        c_j_b = clusters[b,j]
                        loss,ne_loss = self.cal_loss(c_i_b, c_j_b)
                        # 累加每个batch的损失
                        total_loss += loss + alpha * ne_loss
        
        return total_loss / B

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    
