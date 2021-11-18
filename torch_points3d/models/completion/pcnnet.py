import logging
import torch_geometric
from omegaconf import OmegaConf
from chamferdist import ChamferDistance as CD
from torch_points3d.core.losses.completion_loss import ChamferDistance, EarthMoverDistance
from torch_points3d.core.common_modules.base_modules import *
from torch_points3d.utils.config import ConvolutionFormatFactory
from torch_points3d.models.base_model import BaseModel
from torch_points3d.modules.PCN import PcnMultistageEncoder, PcnMultistageDecoder

log = logging.getLogger(__name__)


class PcnNet(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super().__init__(option)

        self._opt = OmegaConf.to_container(option)
        self.alpha = self._opt.get("alpha", 0.1)
        self._is_dense = ConvolutionFormatFactory.check_is_dense_format(self.conv_type)
        self.data_visual = None
        self.loss_coarse = None
        self.loss_fine = None
        self.loss = None

        self.encoder = self.decoder = None
        self._build_model()

        self.is_fine_ouput = self.decoder._apply_fine_decoder

        self.losses = []
        if self._opt.get("coarse_loss", "EMD") == "CD":
            self.loss_func_coarse = ChamferDistance()
        else:
            self.loss_func_coarse = EarthMoverDistance(max_iter=100)

        self.loss_names = ["loss", "loss_coarse"]
        self.visual_names = ["data_visual"]

        if self.is_fine_ouput:
            self.loss_func_fine = ChamferDistance()
            self.loss_names.append("loss_fine")

    def set_input(self, data, device):
        data = data.to(device)
        self.input = data

        # convert from pytorch geometric graph data to standard pytorch tensor with dim BxNx3
        pos, _ = torch_geometric.utils.to_dense_batch(data.pos, data.batch)
        if data.x is not None:
            x, _ = torch_geometric.utils.to_dense_batch(data.x, data.batch)
            self.input_features = torch.cat([pos, x], axis=-1)
        else:
            self.input_features = pos
        if data.y is not None:
            self.ground_truth, _ = torch_geometric.utils.to_dense_batch(data.y, data.batch)
        else:
            self.ground_truth = None
        if not hasattr(data, "batch"):
            self.batch_idx = torch.zeros(self.ground_truth.shape[0]).long()
        else:
            self.batch_idx = data.batch

    def _build_model(self):
        self.encoder = PcnMultistageEncoder(self._opt)
        self.decoder = PcnMultistageDecoder(self._opt)

    def forward(self, *args, **kwargs):
        ### Encode
        glob_feat_v = self.encoder(self.input_features)

        ### Decode
        y_fine, y_coarse = self.decoder(glob_feat_v)

        # sample from ground truth points
        batch_size, num_points, _ = self.ground_truth.shape
        sample_indices = torch.randint(0, num_points, [y_coarse.size(1)])
        ground_truth_coarse = self.ground_truth[:, sample_indices]

        # calculate corase loss
        self.loss_coarse = self.loss_func_coarse(y_coarse, ground_truth_coarse)

        # calculate fine point cloud loss
        if self.is_fine_ouput:
            self.loss_fine = self.loss_func_fine(y_fine, self.ground_truth)
            self.loss = self.loss_coarse + self.alpha * self.loss_fine
        else:
            self.loss = self.loss_coarse

        if self.is_fine_ouput:
            pred_fine = y_fine
        else:
            pred_fine = None

        self.output = dict(x=self.input_features, y=self.ground_truth, pred_coarse=y_coarse, pred_fine=pred_fine)
        self.data_visual = self.output
        return self.output

    def backward(self):
        self.loss.backward()



