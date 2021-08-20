import os
import os.path as osp
import shutil
import json
import timeit

from tqdm.auto import tqdm as tq
from itertools import repeat, product
import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import read_txt_array
import torch_geometric.transforms as T
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.metrics.shapenet_part_tracker import ShapenetPartTracker
from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.utils.download import download_url
from torch_points3d.utils.io_utils import load_h5


class Completion3D(InMemoryDataset):
    r"""The Completion3d point cloud completion dataset from the `"PCN: Point Completion Network"
    <https://arxiv.org/abs/1808.00671>`_
    paper. TODO: more description about DS.

    Args:
        root (string): Root directory where the dataset should be saved.
        categories (string or [string], optional): The category of the CAD
            models (one or a combination of :obj:`"Airplane"`, :obj:`"Bag"`,
            :obj:`"Cap"`, :obj:`"Car"`, :obj:`"Chair"`, :obj:`"Earphone"`,
            :obj:`"Guitar"`, :obj:`"Knife"`, :obj:`"Lamp"`, :obj:`"Laptop"`,
            :obj:`"Motorbike"`, :obj:`"Mug"`, :obj:`"Pistol"`, :obj:`"Rocket"`,
            :obj:`"Skateboard"`, :obj:`"Table"`).
            Can be explicitly set to :obj:`None` to load all categories.
            (default: :obj:`None`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"trainval"`, loads the training and validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"trainval"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = "http://download.cs.stanford.edu/downloads/completion3d/" "dataset2019.zip"

    category_ids = {
        "Airplane": "02691156",
        "Cabinet": "02933112",
        "Car": "02958343",
        "Chair": "03001627",
        "Lamp": "03636649",
        "Couch": "04256520",
        "Table": "04379243",
        "Vessel": "04530566"
    }

    def __init__(
            self,
            root,
            categories=None,
            split="trainval",
            transform=None,
            pre_transform=None,
            pre_filter=None,
            is_test=False,
    ):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        self.is_test = is_test
        
        super(Completion3D, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_raw_paths(self):
        cats = "_".join([cat[:3].lower() for cat in self.categories])
        processed_raw_paths = [os.path.join(self.processed_dir, "raw_{}_{}".format(
            cats, s)) for s in ["train", "val", "test", "trainval"]]
        return processed_raw_paths

    @property
    def raw_file_names(self):
        return ["train", "val", "test", "train.list", "val.list", "test.list"]

    @property
    def processed_file_names(self):
        cats = "_".join([cat[:3].lower() for cat in self.categories])
        return [os.path.join("{}_{}.pt".format(cats, split)) for split in ["train", "val", "test", "trainval"]]

    def download(self):
        if self.is_test:
            return

        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        name = self.url.split("/")[-1].split(".")[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def _process_filenames(self, filenames, split):
        data_raw_list = []
        data_list = []
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}

        has_pre_transform = self.pre_transform is not None

        filenames = [name for name in filenames if name.split(osp.sep)[0] in categories_ids]
        id_scan = -1

        for name in tq(filenames):
            cat = name.split(osp.sep)[0]
            id_scan += 1
            path_ = osp.join(self.raw_dir, split, "{}")
            path_gt = osp.join(path_.format('gt'), f"{name}.h5")
            path_partial = path_gt.replace('gt', 'partial')
            # read gt and input data
            data_gt = load_h5(path_gt)
            data_partial = load_h5(path_partial)

            # assign data
            pos = data_partial
            x = data_partial
            y = data_gt
            category = torch.ones(data_partial.shape[0], dtype=torch.long) * cat_idx[cat]
            id_scan_tensor = torch.from_numpy(np.asarray([id_scan])).clone()
            # create data
            data = Data(pos=pos, x=x, y=y, category=category, id_scan=id_scan_tensor)
            data = SaveOriginalPosId()(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            data_raw_list.append(data.clone() if has_pre_transform else data)
            if has_pre_transform:
                data = self.pre_transform(data)
                data_list.append(data)

        if not has_pre_transform:
            return [], data_raw_list
        return data_raw_list, data_list

    def process(self):
        if self.is_test:
            return

        raw_trainval = []
        trainval = []
        for i, split in enumerate(["train", "val", "test"]):
            path = osp.join(self.raw_dir, f"{split}.list")
            with open(path, "r") as f:
                filenames = f.read().splitlines()

            data_raw_list, data_list = self._process_filenames(sorted(filenames), split)
            if split == "train" or split == "val":
                if len(data_raw_list) > 0:
                    raw_trainval.append(data_raw_list)
                trainval.append(data_list)

            self._save_data_list(data_list, self.processed_paths[i])
            self._save_data_list(
                data_raw_list, self.processed_raw_paths[i], save_bool=len(data_raw_list) > 0)

        self._save_data_list(self._re_index_trainval(
            trainval), self.processed_paths[3])
        self._save_data_list(self._re_index_trainval(
            raw_trainval), self.processed_raw_paths[3], save_bool=len(raw_trainval) > 0)


class Completion3dDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super(Completion3dDataset, self).__init__(dataset_opt)
        try:
            self._category = dataset_opt.category
            is_test = dataset_opt.get("is_test", False)
        except KeyError:
            self._category = None

        self.train_dataset = Completion3D(
            self._data_path,
            self._category,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            is_test=is_test
        )