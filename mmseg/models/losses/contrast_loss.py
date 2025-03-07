# -*- coding:utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
# import torch.autograd as autograd


def mask_type_transfer(mask):
    mask = mask.type(torch.bool)
    # mask = mask.type(torch.uint8)
    return mask


def get_pos_and_neg_mask(bs):
    """Org_NTXentLoss_mask"""
    zeros = torch.zeros((bs, bs), dtype=torch.uint8)
    eye = torch.eye(bs, dtype=torch.uint8)
    pos_mask = torch.cat(
        [
            torch.cat([zeros, eye], dim=0),
            torch.cat([eye, zeros], dim=0),
        ],
        dim=1,
    )
    neg_mask = _get_correlated_mask(bs)
    # (torch.ones(2*bs, 2*bs, dtype=torch.uint8) - torch.eye(2*bs, dtype=torch.uint8))
    pos_mask = mask_type_transfer(pos_mask)
    neg_mask = mask_type_transfer(neg_mask)
    return pos_mask, neg_mask


def _get_correlated_mask(batch_size):
    diag = np.eye(2 * batch_size)
    l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
    l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
    mask = torch.from_numpy((diag + l1 + l2))
    mask = 1 - mask  # .byte()#.type(torch)
    return mask  # .to(self.device)

@LOSSES.register_module()
class NTXentLoss(nn.Module):
    """NTXentLoss

    Args:
        tau: The temperature parameter.
    """

    def __init__(self, bs, tau=0.1, cos_sim=True, eps=1e-8, loss_name='loss_Contrast'):
        super().__init__()
        self.tau = tau
        self.use_cos_sim = cos_sim
        self.eps = eps
        self.bs = bs
        # self.device = device
        self._loss_name = loss_name

        if cos_sim:
            self.cosine_similarity = nn.CosineSimilarity(dim=-1)
            self._loss_name += "_CosSim"

        # Get pos and neg mask
        self.pos_mask, self.neg_mask = get_pos_and_neg_mask(bs)

        # if self.device is not None:
        #     self.pos_mask = self.pos_mask.to(self.device)
        #     self.neg_mask = self.neg_mask.to(self.device)

    def forward_NTXentLoss(self, zi, zj):
        """
        input: {'zi': out_feature_1, 'zj': out_feature_2} (N,C)
        target: one_hot lbl_prob_mat
        """
        zi, zj = F.normalize(zi, dim=1), F.normalize(zj, dim=1)
        bs = zi.shape[0]

        z_all = torch.cat([zi, zj], dim=0)  # input1,input2: z_i,z_j
        # [2*bs, 2*bs] -  pairwise similarity
        if self.use_cos_sim:
            sim_mat = torch.exp(
                self.cosine_similarity(z_all.unsqueeze(1), z_all.unsqueeze(0))
                / self.tau
            )  # s_(i,j)
        else:
            sim_mat = torch.exp(
                torch.mm(z_all, z_all.t().contiguous()) / self.tau
            )  # s_(i,j)

        # pos = torch.sum(sim_mat * self.pos_mask, 1)
        # neg = torch.sum(sim_mat * self.neg_mask, 1)
        # loss = -(torch.mean(torch.log(pos / (pos + neg))))
        sim_pos = sim_mat.masked_select(self.pos_mask).view(2 * bs).clone()

        # [2*bs, 2*bs-1]
        sim_neg = sim_mat.masked_select(self.neg_mask).view(2 * bs, -1)
        # Compute loss
        loss = (-torch.log(sim_pos / (sim_neg.sum(dim=-1) + self.eps))).mean()

        return loss
    
    def forward(self,clusters):
        B,t,_,_ = clusters.shape
        total_loss = []
        for b in range(B):
            # 遍历视图
            for i in range(t):
                for j in range(i+1,t):
                    c_i_b = clusters[b,i] #[N,C]
                    c_j_b = clusters[b,j]
                    loss = self.forward_NTXentLoss(c_i_b, c_j_b)
                    # 累加每个batch的损失
                    total_loss.append(loss)          
        return sum(total_loss) / len(total_loss)
    
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
    

def test():
    contrast_loss = NTXentLoss(3, tau=0.5, cos_sim=True) #这里的bs实际上是样本数目，也就是N
    clusters = torch.randn(2, 4, 3, 256)
    loss = contrast_loss(clusters)
    print(loss)
# test()
