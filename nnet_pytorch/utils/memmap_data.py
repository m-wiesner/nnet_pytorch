#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2020
# Apache 2.0

from datasets.data_utils import memmap_feats, memmap_raw_audio
import pickle
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Takes Kaldi features, converts them to numpy objects and '
        'stores memory-mapped version for efficient access in training.'
    )
    parser.add_argument('feats_scp', help='feats.scp for features, wav.scp for'
        ' raw audio.'
    )
    parser.add_argument('feats_scp_mapped')
    parser.add_argument('metadata')
    parser.add_argument('--utt-list', default=None)
    parser.add_argument('--raw', action='store_true')

    args = parser.parse_args()
    utt_list = []
    if args.utt_list is not None:
        with open(args.utt_list, 'r') as f:
            for line in f:
                utt_list.append(line.strip().split(None, 1)[0])
    memmap_fun = memmap_raw_audio if args.raw else memmap_feats
    utt_lengths, offsets, data_shape = memmap_fun(
        args.feats_scp, args.feats_scp_mapped, utt_list
    ) 
    with open(args.metadata + '.pkl', 'bw') as f:
        pickle.dump([utt_lengths, offsets, data_shape], f)


if __name__ == "__main__":
    main()

