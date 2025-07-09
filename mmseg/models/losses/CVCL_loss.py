# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Union

import torch
import torch.nn as nn
from ..builder import LOSSES

@LOSSES.register_module()
class CVCL_loss(nn.Module):
    def __init__(self,
                num_samples, 
                num_clusters,
                loss_name='CVCL_loss'):
        super().__init__()
        self.num_samples = num_samples
        self.num_clusters = num_clusters

        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self._loss_name = loss_name

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()

        return mask

    def forward_prob(self, q_i, q_j):
        # 传入softmax(dim=-1)((B,N,C))
        B = q_i.size(0)
        entropy = 0
        for b in range(B):
            q_i_b = self.target_distribution(q_i[b])
            q_j_b = self.target_distribution(q_j[b])

            p_i = q_i.sum(0).view(-1)
            p_i /= p_i.sum()
            ne_i = (p_i * torch.log(p_i)).sum()

            p_j = q_j.sum(0).view(-1)
            p_j /= p_j.sum()
            ne_j = (p_j * torch.log(p_j)).sum()

            entropy = ne_i + ne_j

        return entropy/B

    def forward_label(self, q_i, q_j, temperature_l, normalized=False):
        B = q_i.size(0)
        losses = []
        for b in range(B):
            q_i_b = self.target_distribution(q_i[b])
            q_j_b = self.target_distribution(q_j[b])

            q_i_b = q_i_b.t()
            q_j_b = q_j_b.t()
            N = 2 * self.num_clusters
            q = torch.cat((q_i_b, q_j_b), dim=0)

            if normalized:
                sim = (self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l).to(q.device)
            else:
                sim = (torch.matmul(q, q.T) / temperature_l).to(q.device)

            sim_i_j = torch.diag(sim, self.num_clusters)
            sim_j_i = torch.diag(sim, -self.num_clusters)

            positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            mask = self.mask_correlated_samples(N)
            negative_clusters = sim[mask].reshape(N, -1)

            labels = torch.zeros(N).to(positive_clusters.device).long()
            logits = torch.cat((positive_clusters, negative_clusters), dim=1)
            loss = self.criterion(logits, labels)
            losses.append(loss / N)

        return sum(losses)/B


    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

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
