import torch
import torch.nn as nn

from .chamfer_distance import ChamferDistanceFunction
from .emd_distance import emdFunction


class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, pcs1, pcs2):
        """
        Args:
            xyz1: tensor with size of (B, N, 3)
            xyz2: tensor with size of (B, M, 3)
        """
        dist1, dist2 = ChamferDistanceFunction.apply(pcs1, pcs2)  # (B, N), (B, M)
        dist1 = torch.mean(torch.sqrt(dist1))
        dist2 = torch.mean(torch.sqrt(dist2))
        return (dist1 + dist2) / 2


class EarthMoverDistance(nn.Module):
    def __init__(self, eps=0.005, max_iter=3000):
        super(EarthMoverDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, pcs1, pcs2):
        dist, _ = emdFunction.apply(pcs1, pcs2, self.eps, self.max_iter)
        return torch.sqrt(dist).mean()


class ChamferLoss(torch.nn.Module):
    """This is the pure pythonic impl. of chamfer loss, there is no extra cuda-module is needed"""
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    pcs1 = torch.rand(8, 1024, 3)
    pcs2 = torch.rand(8, 1024, 3)

    cd_loss = ChamferDistance()
    print(cd_loss(pcs1, pcs2))

    emd_loss = EarthMoverDistance()
    print(emd_loss(pcs1, pcs2))
