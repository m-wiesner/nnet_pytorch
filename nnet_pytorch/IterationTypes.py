#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2020 
# Apache 2.0

import numpy as np
import torch
import torch.nn.functional as F
import datasets
import sys


def train_epoch(args, generator, model, objective, optim, lr_sched, device='cpu'):
    '''
        The training interator: It takes
             - a minibatch generator
             - a model
             - a training objective
             - an optimizer
             - a learning rate scheduler
        
        It defines how to use these components to perform 1 epoch of training.
    '''
    if args.gpu and args.fp16:
        print("Using fp16 operations")
        scaler = torch.cuda.amp.GradScaler()
        
    total_loss = 0.0
    move_to = datasets.DATASETS[args.datasetname].move_to
    dataset_args = eval(args.datasets)
    total_num_batches = sum(
        [args.batches_per_epoch * ds['num_repeats'] for ds in dataset_args]
    )
    total_num_updates = total_num_batches // args.delay_updates 
   
    for i, b in enumerate(generator, 1*args.delay_updates): 
        b = move_to(b, device)
        print(
            "Iter: ", int(i / args.delay_updates), " of ", total_num_updates,
            "LR: {:0.5e}".format(lr_sched.curr_lr), 
            "bsize: ", b.target.size(0), 
            "cl: ", b.input.size(1), 
            "cw: ", b.input.size(1) - (b.metadata['left_context'] + b.metadata['right_context']),
            end=' '
        )
        if args.gpu and args.fp16:
            with torch.cuda.amp.autocast():
                loss, correct = objective(model, b)
        else:
            loss, correct = objective(model, b)
        if isinstance(loss, int):
            continue;
        print("Loss: {:0.5f}".format(loss.data.item()), end=' ')
        if correct is not None:
            print(" Acc: {:0.5f}".format(float(correct.data.item()) / (b.target.view(-1).size(0))), end=' ')
        total_loss += loss.data.item()
        if args.gpu and args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        loss.detach()
        del b
        # Mimics multigpu training with large batches on a single gpu
        if ((i % args.delay_updates) == 0):
            if args.gpu and args.fp16:
                scaler.unscale_(optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_thresh)
            print("Grad_norm: {:0.5f}".format(grad_norm.data.item()), end='')
            print()
            if args.gpu and args.fp16:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            optim.zero_grad()
            lr_sched.step(1.0)
        else:
            print() 
    return total_loss / args.batches_per_epoch


def validate(args, generator, model, device='cpu'): 
    model.eval()
    move_to = datasets.DATASETS[args.datasetname].move_to
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


def decode_dataset(args, generator, model, device='cpu', output_idx=0):
    '''
        Decoding Iterator: It takes
            - a minibatch generator
            - a model
        and defines how to produce model outputs from the minibatches.        
    '''
    move_to = datasets.DATASETS[args.datasetname].move_to 
    for i, b in enumerate(generator):
        uttname = b.metadata['name'][0]
        b = move_to(b, device)
        model_output = model(b)
        # Chain system
        if 'CrossEntropy' not in args.objective:
            output = model_output[output_idx].clamp(-30, 30)
            lprobs = output.contiguous().view(-1, output.size(2))
        ## XENT
        elif 'CrossEntropy' in args.objective:
            lprobs = F.log_softmax(
                model_output[output_idx], dim=-1
            ).view(-1, model_output[0].size(-1))
        else:
            print("Undefined Objective")
            sys.exit(1)

        yield uttname, lprobs.detach().cpu().numpy()


def decorrupt_dataset(args, generator, model, objective, device='cpu'):
    '''
        Decorruption iterator: It takes
            - a minibatch generator
            - a model
            - a generative objective (with a decorrupt function)
        and returns decorrupted verions of the input.
    '''
    move_to = datasets.DATASETS[args.datasetname].move_to 
    for i, b in enumerate(generator):
        uttname = b.metadata['name'][0]
        targets = None if b.target[0, 0] == -1 else b.target.tolist()
        b = move_to(b, device)
        decorrupt_gen = objective.decorrupt(
            model, b, num_steps=args.num_steps, targets=targets
        )
        # Just yield the first one so we can see what we are started with
        output = b.input.contiguous().view(-1, b.input.size(2))
        output_tgts = b.target.contiguous().view(-1)
        yield uttname, 0, output.detach().cpu(), output_tgts.detach().cpu().tolist()
        for sgld_iter, decorrupted in enumerate(decorrupt_gen, 1):
            output = decorrupted.contiguous().view(-1, decorrupted.size(2))
            output_tgts = b.target.contiguous().view(-1)
            yield uttname, sgld_iter, output.detach().cpu(), output_tgts.detach().cpu().tolist()


def evaluate_energies(args, generator, model, device='cpu'):
    move_to = datasets.DATASETS[args.datasetname].move_to
    for i, b in enumerate(generator, 1):
        b = move_to(b, device)
        model_output = model(b)
        yield model_output[0] 
