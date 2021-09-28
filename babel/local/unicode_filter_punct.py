#!/usr/bin/env python
from __future__ import print_function
import sys
import argparse
import os
import string
import unicodedata

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("keys", help="File with paths to files to process")
    parser.add_argument("text", help="Kaldi format text file output")
    return parser.parse_args()

# Keep Markings such as vowel signs, all letters, and decimal numbers 
VALID_CATEGORIES = ('Mc', 'Mn', 'Ll', 'Lm', 'Lo', 'Lt', 'Lu', 'Nd', 'Zs')


def _filter(s):
    return unicodedata.category(s) in VALID_CATEGORIES


def main():
    args = parse_input()
    odir = os.path.dirname(args.text)
    if not os.path.exists(odir) and odir != "":
        os.makedirs(odir)  

    files = []
    with open(args.keys, "r") as f:
        for l in f:
            files.append(l.strip())

    num_files = len(files)
    print("Number of Files: ", num_files)
    f_num = 1
    with open(args.text, "w", encoding="utf-8") as fo:
        for f in files:
            print("\rFile ", f_num, " of ", num_files, end="")
            text_id = os.path.basename(f).strip(".txt")
            with open(f, "r", encoding="utf-8") as fi:
                utt_num = 0
                for l in fi:
                    l_new = ''.join(
                        [i for i in filter(_filter, l.strip().replace('-', ' '))]
                    ).lower()
                    if l_new.strip() != "":
                        print(u"{}_{:03} {}".format(text_id, utt_num, l_new), file=fo)
                        utt_num += 1
            f_num += 1
    print()

if __name__ == "__main__":
    main()

