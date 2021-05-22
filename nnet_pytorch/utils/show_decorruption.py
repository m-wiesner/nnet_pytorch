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
    args = parser.parse_args()

    files = glob.glob('{}/{}.*.npy'.format(args.idir, args.name))
    files = sorted(files, key=lambda x : int(x.split('.')[-2]))

    images = []
    for f in files:
        fname = os.path.basename(f)
        print(fname)
        out = np.load(f)
        plt.imshow(np.flipud(out.T), vmin=-1.0, vmax=1.0)
        plt.clim(-1, 1)
        plt.colorbar()
        plt.savefig(args.idir + "/" + fname + ".png")
        images.append(imageio.imread(args.idir + "/" + fname + ".png"))
        plt.clf()
    imageio.mimsave(args.ogif, images, duration=0.1)


if __name__ == "__main__":
    main()

