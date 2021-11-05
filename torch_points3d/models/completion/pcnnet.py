import torch.nn.functional as F
import logging
from torch_geometric.data import Data, Batch
from omegaconf import OmegaConf
from chamferdist import ChamferDistance as CD
from torch_points3d.core.losses.completion_losses import ChamferDistance, EarthMoverDistance
from torch_points3d.core.common_modules.base_modules import *
from torch_points3d.utils.config import ConvolutionFormatFactory
from torch_points3d.modules.PointNet import MiniPointNet
from torch_points3d.models.base_model import BaseModel
from torch_points3d.utils.model_building_utils.resolver_utils import flatten_dict
from torch_points3d.datasets.segmentation import IGNORE_LABEL


log = logging.getLogger(__name__)


class PcnNet(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super().__init__(option)

        self._opt = OmegaConf.to_container(option)
        self.embd_dim = self._opt.get('embd_dim', 1024)
        self.coarse_dim = self._opt.get('coarse_dim', 1024)
        self._is_dense = ConvolutionFormatFactory.check_is_dense_format(self.conv_type)
        self.data_visual = None
        self.loss_emd_coarse = None
        self.loss_cd_coarse = None
        self.loss = None

        self._build_model()

        self.loss_names = ["loss", "loss_cd_coarse", "loss_emd_coarse"]

       # self.loss_chamfer = ChamferLoss()
        self.loss_chamfer2 = CD()
        self.loss_chamfer = ChamferDistance()
        self.loss_emd = EarthMoverDistance(max_iter=300)

        self.visual_names = ["data_visual"]

    def set_input(self, data, device):
        data = data.to(device)
        self.input = data
        if data.x is not None:
            self.input_features = torch.cat([data.pos, data.x], axis=-1)
        else:
            self.input_features = data.pos
        if data.y is not None:
            self.labels = data.y
        else:
            self.labels = None
        if not hasattr(data, "batch"):
            self.batch_idx = torch.zeros(self.labels.shape[0]).long()
        else:
            self.batch_idx = data.batch

    def _build_model(self):
        self.encoder_1 = MiniPointNet([3, 128, 256], None, return_local_out=True)
        self.encoder_2 = MiniPointNet([2*256, 512, self.embd_dim], None)

        self.decoder1 = MLP([self.embd_dim, 1024, self.coarse_dim * 3])

    def encode(self):
        # N = the real number of pointes (e.g. N=2048) x batch (e.g. B=16) = 32768
        N = self.input_features.size(0)

        # MLP 1
        # dim(x)=batchsize x feat_length, dim(local_out) = num_points x feat_length
        glob_feat_g, f = self.encoder_1(self.input_features, self.batch_idx)

        batch_size, feat_len = glob_feat_g.size()
        num_points = int(N / batch_size)

        # expand features
        glob_feat_g = glob_feat_g.unsqueeze(1)
        glob_feat_g = glob_feat_g.repeat((1, num_points, 1)).view(-1, feat_len)

        # check the dimensions
        assert glob_feat_g.size(0) == f.size(0)

        # MLP 2
        glob_feat_v = torch.cat([glob_feat_g, f], axis=-1)
        glob_feat_v = self.encoder_2(glob_feat_v, self.batch_idx)
        return glob_feat_v

    def decode(self, x):
        ### Decoder
        y_coarse = self.decoder1(x)
        y_coarse = y_coarse.view(-1, self.coarse_dim, 3)
        return y_coarse

    def forward(self, *args, **kwargs):
        ### Encode
        glob_feat_v = self.encode()

        ### Decode
        y_coarse = self.decode(glob_feat_v)

        batch_size = glob_feat_v.size(0)
        labels = self.labels.view(batch_size, -1, 3)
        if self.labels is not None:
            self.loss_cd_coarse = self.loss_chamfer(y_coarse, labels)
            self.loss_emd_coarse = self.loss_emd(y_coarse, labels)
            self.loss = 0.4 * self.loss_cd_coarse + 0.6 * self.loss_emd_coarse

        out_data = []
        for data in y_coarse:
            out_data.append(Data(pos=data))

        self.output = Batch.from_data_list(out_data)

        self.data_visual = self.input
        self.data_visual.pred = self.output
        return self.output

    def backward(self):
        self.loss.backward()


class ChamferLoss(torch.nn.Module):

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
