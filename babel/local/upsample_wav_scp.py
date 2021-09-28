#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2019  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

from __future__ import print_function
import argparse
import sys
import os



def load_scp(f):
    scp = {}
    for l in f:
        uttid, cmd = l.strip().split(None, 1)
        scp[uttid] = cmd
    return scp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('wavscp')

    args = parser.parse_args()
    with open(args.wavscp) as f:
        scp = load_scp(f)
    for utt in sorted(scp):
        utt_cmd = scp[utt].split()
        if len(utt_cmd) == 8:
            print('{} {} -f wav -p -c 1 {} | sox -R -t wav - -t wav - rate 16000 dither |'.format(
                    utt, utt_cmd[0], utt_cmd[6] 
                )
            )
        elif len(utt_cmd) == 13:
            print('{} sox {} -r 16000 -c 1 -b 16 -t wav - downsample |'.format(utt, utt_cmd[1]))


if __name__ == "__main__":
    main()

