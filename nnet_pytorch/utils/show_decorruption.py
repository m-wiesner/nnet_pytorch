#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2019  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

from __future__ import print_function
import argparse
import sys
import os
import glob
import imageio
from matplotlib import pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('idir')
    parser.add_argument('ogif')
    parser.add_argument('name', type=str)
    parser.add_argument('--initval', type=float, default=1.0)
    parser.add_argument('--dur', type=float, default=0.1)
    parser.add_argument('--every-k', type=int, default=1)
    args = parser.parse_args()

    files = glob.glob('{}/{}.*.npy'.format(args.idir, args.name))
    files = sorted(files, key=lambda x : int(x.split('.')[-2]))

    images = []
    for f in files[::args.every_k]:
        fname = os.path.basename(f)
        print(fname)
        out = np.load(f)
        plt.imshow(np.flipud(out.T), vmin=-args.initval, vmax=args.initval)
        plt.clim(-args.initval, args.initval)
        plt.colorbar()
        plt.savefig(args.idir + "/" + fname + ".png")
        images.append(imageio.imread(args.idir + "/" + fname + ".png"))
        plt.clf()
    imageio.mimsave(args.ogif, images, duration=args.dur)


if __name__ == "__main__":
    main()

