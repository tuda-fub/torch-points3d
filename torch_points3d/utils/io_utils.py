import h5py
import torch


def load_h5(path, key='data', dtype=None, device=None):
    f = h5py.File(path, 'r')
    data = torch.tensor(f[key], dtype=dtype, device=device)
    f.close()
    return data