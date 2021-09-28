#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2019  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

from __future__ import print_function
import argparse
import sys
import os
import random


def main():
    parser = argparse.ArgumentParser("Randomly chooses k utterances from"
        " different langs"
    )
    parser.add_argument('segments',
        help='segments file from kaldi data directory',
        type=str
    )
    parser.add_argument('k', type=int)
    parser.add_argument('chosen')
    args = parser.parse_args()

    lang2utt = {}
    with open(args.segments) as f_segments:
        for l in f_segments:
            lang = l.strip().split()[0][-16:-6]
            utt = l.strip().split()[0]
            if lang not in lang2utt:
                lang2utt[lang] = []
            lang2utt[lang].append(utt)  
    
    k_chosen = 0
    utts = set()
    while len(utts) < args.k:
        lang = random.sample(lang2utt.keys(), 1)[0]
        utts.add(random.sample(lang2utt[lang], 1)[0]) 
    
    with open(args.chosen, 'w') as f_chosen:
        for l in utts:
            print(l, file=f_chosen)  

if __name__ == "__main__":
    main()

