import torch.nn.functional as F
import torch
import numpy as np

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        # self.margin = margin

    def forward(self, euclidean_distance, label, margin):
        # euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class Smooth_contrastive(torch.nn.Module):

    def __init__(self, e=0.001, al=1, lam=1):
        super(Smooth_contrastive,self).__init__()
        self.e = e
        self.alph = al
        self.lam = lam

    def forward(self, diff, pcx, m, ):

        ls = torch.sum(torch.mul(diff, 1 - pcx))/(diff.shape[1]*diff.shape[2])

        sub = torch.sub(m, diff)

        r1 = np.arctan(1 / self.e)  # e is fixed, r1= 1.56
        r2 = torch.atan(-sub / self.e) / r1
        Asub = -sub * r2 - (self.e / 2.0 * r1) * torch.log(1 + torch.pow(sub / self.e, 2))  # case2 approximation

        maxsub = (0.5 * (sub + Asub)) ** 2

        ld = torch.sum(torch.mul(maxsub, pcx))/(diff.shape[1]*diff.shape[2])
        # print('sub:{}, Asub:{}, ls: {}, ld: {}'.
        #       format(torch.sum(sub) / sub.numel(), torch.sum(Asub) / Asub.numel(), ls, ld))

        overal_loss = self.alph * ls + self.lam * ld

        return overal_loss