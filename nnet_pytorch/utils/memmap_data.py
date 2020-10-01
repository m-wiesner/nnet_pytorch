#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2020
# Apache 2.0

from datasets.data_utils import memmap_feats 
import pickle
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Takes Kaldi features, converts them to numpy objects and '
        'stores memory-mapped version for efficient access in training.'
    )
    parser.add_argument('feats_scp')
    parser.add_argument('feats_scp_mapped')
    parser.add_argument('metadata')
    parser.add_argument('--utt-list', default=None)

    args = parser.parse_args()
    utt_list = []
    if args.utt_list is not None:
        with open(args.utt_list, 'r') as f:
            for line in f:
                utt_list.append(line.strip())
    utt_lengths, offsets, data_shape = memmap_feats(
        args.feats_scp, args.feats_scp_mapped, utt_list
    ) 
    with open(args.metadata + '.pkl', 'bw') as f:
        pickle.dump([utt_lengths, offsets, data_shape], f)


if __name__ == "__main__":
    main()

