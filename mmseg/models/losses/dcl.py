import numpy as np
import torch
import torch.nn as nn
from ..builder import LOSSES
import torch.nn.functional as F
SMALL_NUM = np.log(1e-45)

@LOSSES.register_module()
class DCL(nn.Module):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, 
                 temperature=0.1, 
                 weight_fn=None,
                 loss_name= 'loss_dcl'):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
        self._loss_name = loss_name


    def forward_dcl(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()
    
    def forward(self,clusters):
        B,t,_,_ = clusters.shape
        total_loss = []
        for b in range(B):
            # 遍历视图
            for i in range(t):
                for j in range(i+1,t):
                    c_i_b = clusters[b,i] #[N,C]
                    c_j_b = clusters[b,j]
                    loss = self.forward_dcl(c_i_b, c_j_b) + self.forward_dcl(c_j_b, c_i_b)
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
    
@LOSSES.register_module()
class DCLW(DCL):
    """
    Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, sigma=0.5, temperature=0.1):
        weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        super(DCLW, self).__init__(weight_fn=weight_fn, temperature=temperature, loss_name='loss_dclw')


def test():
    dcl = DCL(temperature=0.1)
    dclw = DCLW(temperature=0.1)
    clusters = torch.randn(2, 4, 10, 128)
    loss = dcl(clusters)
    print(loss)

    loss = dclw(clusters)
    print(loss)

# test()