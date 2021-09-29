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
    parser.add_argument('odir',
        help='',
        type=str
    )
    parser.add_argument('dicts', nargs='+')
    args = parser.parse_args()

    # File showing for each lexicon which map file to use from words to ints
    dicts = args.dicts
    with open(os.path.join(args.odir, 'word_maps.scp'), 'w') as f:
        for i, l in enumerate(dicts):
            print(l, 'wordmap.{}'.format(i), file=f)
    
    # Gather silence phonemes across all dicts
    silence_phones = gather_silence_phones(dicts)
   
    sil_words = set()
    word_maps = {d: {} for d in dicts}
    word_count = 1
    lexicon = {} 
    
    for d in dicts:
        with open(os.path.join(d, 'lexicon.txt'), encoding='utf-8') as f:
            for l in f:
                word, pron = l.strip().split(None, 1)
                if is_silence_word(l, silence_phones):
                    sil_words.add((word, pron))
                else:
                    if word not in word_maps[d]:
                        word_maps[d][word] = word_count  
                        word_count += 1
                    if word_maps[d][word] not in lexicon:
                        lexicon[word_maps[d][word]] = []
                    lexicon[word_maps[d][word]].append(pron)
    
    for w in sil_words:
        if w[0] not in lexicon:
            lexicon[w[0]] = []
        lexicon[w[0]].append(w[1])
        for d in word_maps:
            word_maps[d][w[0]] = word_count
        word_count += 1

    with open(os.path.join(args.odir, 'lexicon.txt'), 'w', encoding='utf-8') as f_lexicon:
        for w in lexicon:
            for pron in lexicon[w]:
                print(w, pron, file=f_lexicon)
    
    for i, d in enumerate(dicts):
        fname = 'wordmap.{}'.format(i)
        with open(os.path.join(args.odir, fname), 'w') as f_word_map:
           for w in word_maps[d]:
               print(w, word_maps[d][w], file=f_word_map)
    

def gather_silence_phones(dicts):
    silence_phones = set()
    for d in dicts:
        with open(os.path.join(d, 'silence_phones.txt')) as f:      
            for l in f:
                silence_phones.add(l.strip())
    return silence_phones


def is_silence_word(lexicon_entry, silence_phones):
    return lexicon_entry.strip().split(None, 1)[1] in silence_phones 
 

if __name__ == "__main__":
    main()

