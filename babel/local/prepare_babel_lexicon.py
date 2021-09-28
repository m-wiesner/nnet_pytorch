#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2019  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

from __future__ import print_function
import argparse
import sys
import os
import unicodedata
import re

VOWELS = [
    '6',
    'I',
    'I\\',
    '}',
    'Y',
    'E',
    '{',
    '@\\',
    'e',
    'M',
    '7',
    'O',
    '1',
    'V',
    'Q',
    '9',
    '3\\',
    '3',
    '&',
    'U',
    'A',
    'y',
    '8',
    'U\\',
    '2',
    'a',
    'u',
    'i',
    'o',
    '@',
    '@`',
]

MODIFIERS = [
    '~',
    '_~',
    ':',
]

TAGS = [
    '~',
    'n',
    'w',
    'h',
    'j',
    "'",
    '=',
    '>',
    '<',
    'd',
    'o',
    '+',
    '"',
    '-',
    '0',
    'a',
    'e',
]

VOWELS_REGEX = '(' + '|'.join([i.replace('\\', '\\\\') + '(' + '|'.join(MODIFIERS) + ')?' for i in sorted(VOWELS, key=lambda x: len(x), reverse=True)]) + ')'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lexicon')

    args = parser.parse_args()

    with open(args.lexicon, encoding='utf-8') as f_lexicon:
        lexicon = parse_lexicon(f_lexicon)

    for w in sorted(lexicon.keys()):
        for pron in lexicon[w]:
            print(w, end='\t')
            new_pron_list = []
            for p in pron.split():  
                if re.match(VOWELS_REGEX, p) is None:
                    new_pron_list.append(p)
                else:      
                    p_new = ' '.join(i[0] for i in re.findall(VOWELS_REGEX, p))
                    new_pron_list.append(p_new)
            new_pron = ' '.join(new_pron_list)
            new_pron = re.sub(r":", r"_:", new_pron) # Treat length as optional feature on which to split phonemes
            new_pron = re.sub(r"'", r'_j', new_pron) # ' --> _j depending on the XSAMPA representation
            new_pron = re.sub(r"gj", r'J\\', new_pron) # gj --> J\ (Turkish uses gj for some reason)
            new_pron = re.sub(r"Hi", r'H i', new_pron) # Hi --> H i (Haitina Diphthong with H which is not listed as a vowel in the
                                                        #           XSAMPA wikipedia entry)
            new_pron = re.sub(r"_hj", r"_h_j", new_pron) # _hj --> _h_j
            new_pron = re.sub(r"_cj", r"_c_j", new_pron) # _cj --> _c_j
            new_pron = re.sub(r"~", r"_~", new_pron) # ~ --> _~
            new_pron = re.sub(r"kw", r"k_w", new_pron) # {k,g}w --> {k,g}_w (Cantonese consonants)
            new_pron = re.sub(r"kw", r"k_w", new_pron)
            new_pron = re.sub(r"n=", r"n_=", new_pron) # n= --> n_= (Swahili)
            new_pron = re.sub(r"n_dZ", r"dZ_n", new_pron) # prenasal dZ (Swahili)
            new_pron = re.sub(r"n_d", r"d_n", new_pron) # prenasal d (Swahili)
            new_pron = re.sub(r"m_b", r"b_m", new_pron) # prelabialized b (Swahili)
            new_pron = re.sub(r"m=", r"m_=", new_pron) # m= --> m_= (Swahili)
            new_pron = re.sub(r"N=", r"N_=", new_pron) # N= --> N_= (Swahili)
            new_pron = re.sub(r"N_g", r"g_N", new_pron) # prenasal g
            #new_pron = re.sub(r"g_!\_t", r"!\_t_g", new_pron) 
            new_pron = re.sub(r"g_!\\_t", r"!\_t_g", new_pron) # Swalihi clicks
            #new_pron = re.sub(r"g_|\_t", r"|\_t_g", new_pron)
            new_pron = re.sub(r"g_\|\\_t", r"|\_t_g", new_pron)
            new_pron = re.sub(r"g_\|\\\|\\_t", r"|\\|\_t_g", new_pron)
            #new_pron = re.sub(r"g_|\|\_t", r"|\|\_t_g", new_pron) 
            new_pron = re.sub(r"_[RF]", r"", new_pron) # Lithuanian writes tone differently (strip tone _R _F)
            new_pron = re.sub(r"n_D", r"D_n", new_pron) # prenasal D (dholuo)
            new_pron = re.sub(r"([^ _]+)w", r"\1_w", new_pron) # Labialized consonants in Amharic and Igbo
            new_pron = re.sub(r"([^ _]+)>", r"\1_>", new_pron) # Ejective consonants in Amharic and Igbo
            new_pron = re.sub(r"([^ _]+)j", r"\1_j", new_pron) # Palatal consonants in Igbo
            new_pron = re.sub(r"([^ _]+)n", r"\1_~", new_pron) # Nasal consonants in Igbo
            new_pron = re.sub(r"([^ _]+)h", r"\1_h", new_pron) # Aspirated consonants in Igbo
            new_pron = re.sub(r"kp", r"p_<_1", new_pron) # Pronunciation variant of p_< in Igbo dialect
            new_pron = re.sub(r"gb", r"g_<_1", new_pron) # Pronunciation variant of g_< in Igbo dialect
            print(new_pron)

def parse_lexicon(f):
    # Figure out how many fields
    elements = {}
    pron_start_idx = 1
    for l in f:
        element = l.strip().split('\t')
        elements[element[0]] = element[1:]
        if len(element[1:]) < 2:
            pron_start_idx = 0

    lexicon = {}
    for word in elements:
        for pron in elements[word][pron_start_idx:]:
            pron = ' '.join(re.sub('[#."%]| _[0-9RF]', '', pron).split())
            if word not in lexicon:
                lexicon[word] = []
            lexicon[word].append(pron)
    return lexicon


def parse_tags(pron):
    tags = re.compile


if __name__ == "__main__":
    main()


