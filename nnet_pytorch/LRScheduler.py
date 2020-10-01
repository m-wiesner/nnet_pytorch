#-*- coding: utf-8 -*-
# Copyright 2020
# Apache 2.0

from __future__ import print_function
import sys
import os
import math


class LRScheduler(object):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--warmup', type=int, default=0)
        parser.add_argument('--decay', type=float, default=0.0)
        parser.add_argument('--fixed', type=int, default=0)

    def __init__(self, optimizer, args):
        self.optimizer = optimizer
        self.warmup = args.warmup
        self.fixed = args.fixed
        self.decay = args.decay
        
        self.num_warmup_updates = 0
        self.num_fixed_updates = 0
        self.num_decay_updates = 0
        self.lr = self.optimizer.param_groups[0]['lr']  
        if self.warmup > 0:
            self.set_lr(0.000000001)
            self.curr_lr = 0.000000001
        else:
            self.curr_lr = self.lr

    def step(self, num_new_updates):
        if self.warmup > 0 and self.num_warmup_updates < self.warmup:
            self.num_warmup_updates += num_new_updates 
            slope = self.lr / float(self.warmup) 
            new_lr = slope * self.num_warmup_updates
        elif self.fixed > 0 and self.num_fixed_updates < self.fixed:
            self.num_fixed_updates += num_new_updates
            new_lr = self.lr 
        else:
            self.num_decay_updates += num_new_updates
            factor = math.exp(-self.decay * self.num_decay_updates) 
            new_lr = self.lr * factor
        self.set_lr(new_lr)
        self.curr_lr = new_lr
    
    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return {
            'warmup': self.warmup,
            'fixed': self.fixed,
            'decay': self.decay,
            'warmup_updates': self.num_warmup_updates,
            'fixed_updates': self.num_fixed_updates,
            'decay_updates': self.num_decay_updates,
            'lr': self.lr,
            'curr_lr': self.curr_lr,
        }
    
    def load_state_dict(self, state_dict):
        self.warmup = state_dict['warmup']
        self.fixed = state_dict['fixed']
        self.decay = state_dict['decay']
        self.num_warmup_updates = state_dict['warmup_updates']
        self.num_fixed_updates = state_dict['fixed_updates']
        self.num_decay_updates = state_dict['decay_updates']
        self.lr = state_dict['lr']
        self.curr_lr = state_dict['curr_lr']
