#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2020
# Apache 2.0

from data_utils import memmap_feats 
import pickle
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Takes Kaldi features, converts them to numpy objects and
        stores memory-mapped version for efficient access in training.'
    )
    parser.add_argument('feats_scp')
    parser.add_argument('feats_scp_mapped')

    args = parser.parse_args()
    utt_lengths, offsets, data_shape = memmap_feats(
        args.feats_scp, args.feats_scp_mapped
    ) 
    with open(args.feats_scp + '.pkl', 'bw') as f:
        pickle.dump([utt_lengths, offsets, data_shape], f)


if __name__ == "__main__":
    main()

