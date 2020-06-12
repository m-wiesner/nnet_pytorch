#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2019  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

from __future__ import print_function
import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('utt2num_frames',
        help='',
        type=str
    )
    parser.add_argument('--subsample', type=int, default=1)

    args = parser.parse_args()

    with open(args.utt2num_frames, 'r') as f:
        for l in f:
            utt, frames = l.strip().split(None, 1)
            print(utt, end='')
            num_frames = len(range(0, int(frames), args.subsample))
            print(' -1' * num_frames)

if __name__ == "__main__":
    main()

