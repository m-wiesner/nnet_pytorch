#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This script creates paragraph level text file. It reads 
    the line level text file and combines them to get
    paragraph level file.
  Eg. local/combine_line_txt_to_paragraph.py
  Eg. Input:  103085_w5Jyq3XMbb3WwiKQ_0002 यहाँ हम अपने ऑपरेटिंग सिस्टम के रूप में gnu/लिनक्स और लिबर ऑफिस वर्जन 334 का उपयोग कर
              103085_w5Jyq3XMbb3WwiKQ_0003 चलिए अपनी प्रस्तुति sample impress open करते हैं जिसे पिछले tutorial में
              103085_w5Jyq3XMbb3WwiKQ_0004 चलिए देखते हैं कि screen पर क्या क्या है
      Output: 103085_w5Jyq3XMbb3WwiKQ याँ हम अपने ऑपरेिं ... चलिए देखते हैं कि screen पर क्या क्या
"""

import os
import io
import sys


def _load_text(f):
    paragraph_txt_dict = dict()
    for line in f:
        try:
            uttid, text = line.strip().split(None, 1)
        except ValueError:
            uttid = line.strip()
            text = ''
        sequence_id = int(uttid.split('_')[-1])
        recoid = uttid.split('_')[1]
        if recoid not in paragraph_txt_dict:
            paragraph_txt_dict[recoid] = dict()
        paragraph_txt_dict[recoid][sequence_id] = text
    return paragraph_txt_dict


def main(infile, output): 
    paragraph_txt_dict = _load_text(infile)
    for para_id, sequence_dict in sorted(paragraph_txt_dict.items(), key=lambda x: x[0]):
        new_text = [] 
        for line_id, text in sorted(sequence_dict.items(), key=lambda x: x[0]):
            new_text.append(text.strip())
        print('{} {}'.format(para_id, ' '.join(new_text)), file=output)


if __name__ == "__main__":
    infile = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main(infile, output)
