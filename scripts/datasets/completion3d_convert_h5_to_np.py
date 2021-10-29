import os
import os.path as osp
import glob
import threading
from multiprocessing import Pool

from tqdm.auto import tqdm as tq
import argparse
import numpy as np
from torch_points3d.utils.io_utils import load_h5


def find_largest_divisor(num, max_divisor, min_divisor=1):
    for i in range(max_divisor, min_divisor, -1):
        if num % i == 0:
            return i


def convert_h5_to_npy(fpath):
    h5file = load_h5(fpath)
    data = h5file.data
    fname = osp.basename(fpath).split('.')[0]
    fbase = osp.dirname(fpath)
    np.save(f"{fbase}/{fname}.npy", data)


def convert_batch(fbatch):
    n = len(fbatch)
    with Pool(n) as p:
        p.map(convert_h5_to_npy, fbatch)


def main(args):
    file_paths = glob.glob(f"{args.path}/**/*.h5", recursive=True)
    divisor = find_largest_divisor(len(file_paths), 10)
    file_bins = np.array(file_paths).reshape(-1, divisor)

    print("Starting converting ... ")
    for i in tq(range(len(file_bins))):
        convert_batch(file_bins[i])

    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help="Root directory of the h5 files")

    args = parser.parse_args()
    main(args)
