#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2020
# Apache 2.0

from __future__ import print_function
import argparse
import sys
import os
import json
import torch
import models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modeldir',
        help='Output model directory',
        type=str,
    )
    parser.add_argument('idim', type=int)
    parser.add_argument('start',
        help='Start epoch model to average',
        type=int,
    )
    parser.add_argument('end',
        help='End epoch model to average',
        type=int,
    )
    #parser.add_argument('--weights',
    #    help='Weights for each model',
    #)
    args =  parser.parse_args()

    conf = json.load(open(args.modeldir + '/conf.json'))
    conf['idim'] = args.idim
    new_model = models.MODELS[conf['model']].build_model(conf)
    new_dict = new_model.state_dict()
    for name, param in new_dict.items():
        param.mul_(0.0)
    
    fraction = 1.0 / (args.end - args.start + 1)
    for m in range(args.start, args.end + 1):
        state_dict = torch.load(
            args.modeldir + '/{}.mdl'.format(m),
            map_location=torch.device('cpu')
        )
        for name, p in state_dict['model'].items():
            if name in new_dict:
                new_dict[name].add_(fraction, p)
    torch.save(
        {'model': new_dict},
        args.modeldir + '/{}_{}.mdl'.format(args.start, args.end)
    ) 
   
if __name__ == "__main__":
    main()

