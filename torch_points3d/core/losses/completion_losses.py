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