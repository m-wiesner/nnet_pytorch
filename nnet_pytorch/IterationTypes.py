#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2020 
# Apache 2.0

import numpy as np
import torch
import torch.nn.functional as F
from data_utils import batches, validation_batches, evaluation_batches, move_to 


def train_epoch(args, generator, model, objective, optim, lr_sched, device='cpu'):
    total_loss = 0.0
    datasets = eval(args.datasets)
    for i, b in enumerate(generator, 1): 
        b = move_to(b, device)
        loss, correct = objective(model, b)
        if isinstance(loss, int):
            continue;
        print(
            "Iter: ", i, " of ", args.batches_per_epoch * len(datasets),
            "Loss: ", loss.data.item(),
            "LR: ", lr_sched.curr_lr, end=' '    
        )
        if correct is not None:
            print(" Acc: ", float(correct.data.item()) / (b.target.view(-1).size(0)), end='')
        print()
        total_loss += loss.data.item()
        loss.backward()
        loss.detach() 
        # Mimics multigpu training with large batches on a single gpu
        if ((i % args.delay_updates) == 0):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_thresh)
            optim.step()
            optim.zero_grad()
            lr_sched.step(1.0 / len(datasets)) 
    return total_loss / args.batches_per_epoch


def validate(args, generator, model, device='cpu'): 
    model.eval()
    with torch.no_grad():
        correct = 0.0
        avg_loss = 0.0
        num_tokens = 0.0
        for i, b in enumerate(generator):
            b = move_to(b, device)
            output = model(b)[0]
            lprobs = F.log_softmax(output, dim=-1)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            lprobs = lprobs[:b.target.view(-1).size(0), :]
            loss = F.nll_loss(lprobs, b.target.view(-1), reduction='sum')
            avg_loss += loss.data.item()
            correct += torch.sum(lprobs.argmax(1) == b.target.view(-1))
            num_tokens += lprobs.size(0)
        avg_loss /= num_tokens
        correct = 0 if num_tokens == 0 else float(correct.data.item()) / num_tokens
        print()
    model.train()
    return avg_loss, correct


def decode_dataset(args, generator, model, device='cpu'): 
    for i, b in enumerate(generator):
        uttname = b.metadata['name'][0]
        b = move_to(b, device)
        model_output = model(b)
        # Chain system
        if 'LFMMI' in args.objective:
            output = model_output[0].clamp(-30, 30)
            lprobs = output.contiguous().view(-1, output.size(2))
        ## XENT
        elif 'CrossEntropy' in args.objective:
            lprobs = F.log_softmax(
                model_output[0], dim=-1
            ).view(-1, model_output[0].size(-1))

        yield uttname, lprobs.detach().cpu().numpy()


def evaluate_energies(args, generator, model, device='cpu'):
    for i, b in enumerate(generator, 1):
        b = move_to(b, device)
        model_output = model(b)
        yield model_output.data.item() 
          
