from torch_points3d.core.common_modules.base_modules import *
from torch_points3d.modules.PointNet import MiniPointNet


class PcnMultistageEncoder(nn.Module):
    def __init__(self, opt):
        super(PcnMultistageEncoder, self).__init__()
        self.embd_dim = opt.get('embd_dim', 1024)

        self.encoder_1 = MiniPointNet([3, 128, 256], None, return_local_out=True)
        self.encoder_2 = MiniPointNet([2 * 256, 512, self.embd_dim], None)

    def forward(self, x):
        # dim(x) = BxNx3
        glob_feat_g, point_feat = self.encoder_1(x)

        batch_size, feat_len = glob_feat_g.size()
        num_points = x.size(1)

        # expand features
        glob_feat_g = glob_feat_g.unsqueeze(1)
        glob_feat_g = glob_feat_g.repeat((1, num_points, 1))

        # check the dimensions
        assert glob_feat_g.size(1) == point_feat.size(1)

        # MLP 2
        glob_feat_v = torch.cat([glob_feat_g, point_feat], axis=-1)
        glob_feat_v = self.encoder_2(glob_feat_v)
        return glob_feat_v


class PcnMultistageDecoder(nn.Module):
    def __init__(self, opt):
        super(PcnMultistageDecoder, self).__init__()
        self.embd_dim = opt.get('embd_dim', 1024)
        self.coarse_dim = opt.get('coarse_dim', 1024)
        self.grid_size = opt.get('grid_size', 4)
        self.grid_scale = opt.get('grid_scale', 0.05)
        self._apply_fine_decoder = opt.get('fine_decoder', True)

        self.fine_dim = self.grid_size ** 2 * self.coarse_dim

        self.y_coarse = None
        self.y_fine = None

        self._decoder_coarse = nn.Sequential(
            MLP([self.embd_dim, 1024, 1024]),
            nn.Linear(1024, self.coarse_dim * 3)
        )

        if self._apply_fine_decoder:
            # 2 := dim of the grid array (16x2)
            # 3 := dim of coarse output (Nx3)
            # 1024 := dim of global feat "v"
            in_channels = self.coarse_dim + 3 + 2          # 1029
            self._decoder_fine = nn.Sequential(
                MLP([in_channels, 512, 512]),
                nn.Linear(512, 3)
            )

    def fine_decode(self, x):
        global_feats = x
        batch_size = x.shape[0]

        # global features
        global_feats = global_feats.unsqueeze(1).repeat(1, self.fine_dim, 1)  # Bx16384x1024

        # 2D grid and grid feature
        grid_space = torch.linspace(-self.grid_scale, self.grid_scale, steps=self.grid_size).to(global_feats.device)
        grid = torch.meshgrid(grid_space, grid_space)
        grid = torch.stack(grid, dim=2).reshape(-1, 2)
        grid = grid.unsqueeze(0)
        grid_feats = grid.repeat([batch_size, self.coarse_dim, 1])

        # coarse point cloud feature
        y_coarse_feats = self.y_coarse.unsqueeze(2)
        y_coarse_feats = y_coarse_feats.repeat(1, 1, self.grid_size ** 2, 1)  # Bx1024x16x3
        y_coarse_feats = y_coarse_feats.reshape(-1, self.fine_dim, 3)

        features = torch.cat([global_feats, grid_feats, y_coarse_feats], dim=2)

        center_points = self.y_coarse.unsqueeze(2)
        center_points = center_points.repeat(1, 1, self.grid_size ** 2, 1)  # Bx1024x16x3
        center_points = center_points.reshape(-1, self.fine_dim, 3)

        self.y_fine = self._decoder_fine(features)
        self.y_fine = self.y_fine + center_points

    def forward(self, x):
        # coarse point cloud
        self.y_coarse = self._decoder_coarse(x)
        self.y_coarse = self.y_coarse.view(-1, self.coarse_dim, 3)

        if self._apply_fine_decoder:
            self.fine_decode(x)
            return self.y_fine, self.y_coarse
        return None, self.y_coarse

