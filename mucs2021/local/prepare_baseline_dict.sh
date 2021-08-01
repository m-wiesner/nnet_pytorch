#!/bin/bash

lang=hindi

. ./utils/parse_options.sh

bengali_dict=https://raw.githubusercontent.com/navana-tech/baseline_recipe_is21s_indic_asr_challenge/master/is21-subtask2-kaldi/bengali_baseline/corpus/lang
hindi_dict=https://raw.githubusercontent.com/navana-tech/baseline_recipe_is21s_indic_asr_challenge/master/is21-subtask2-kaldi/hindi_baseline/corpus/lang

if [[ $lang = "hindi" ]]; then
  dict=$hindi_dict
elif [[ $lang = "bengali" ]]; then
  dict=$bengali_dict
fi

wget ${dict}/lexicon.txt -O data/dict/lexicon.txt
wget ${dict}/nonsilence_phones.txt -O data/dict/nonsilence_phones.txt
wget ${dict}/silence_phones.txt -O data/dict/silence_phones.txt
wget ${dict}/optional_silence.txt -O data/dict/optional_silence.txt
