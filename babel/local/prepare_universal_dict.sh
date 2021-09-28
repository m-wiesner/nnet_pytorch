#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

###########################################################################
# Create dictionaries with split diphthongs and standardized tones
# This script recreates the dictionary directories by modifying the
# the phonemic inventory of the languages.
# All diphthongs and triphthongs are split into their constituent phones when
# possible, all tone markings, which have no standard representation across
# languages in the x-sampa phoneme set, are ignored. Features corresponding to
# IPA diacritics are mapped to kaldi tags (_*) such that they may have state
# tied parameters.
###########################################################################

. ./path.sh

dict=data/dict_universal
src=data/local
istest=false

. ./utils/parse_options.sh
if [ $# -ne 1 ]; then
  echo >&2 "Usage: ./local/prepare_dictionary.sh --dict data/dict_universal <lang_id>"
  exit 1
fi 

l=$1

mkdir -p $dict

echo "Making dictionary for ${l}"

# Create silence lexicon (This is the set of non-silence phones used in the
# babel recipes
echo -e "<silence>\tSIL\n<unk>\t<oov>\n<noise>\t<sss>\n<v-noise>\t<vns>" \
  > ${dict}/silence_lexicon.txt

# Create non-silence lexicon
grep -vFf ${dict}/silence_lexicon.txt ${src}/filtered_lexicon.txt \
  > ${src}/nonsilence_lexicon.txt

LC_ALL= python local/prepare_babel_lexicon.py ${src}/nonsilence_lexicon.txt > ${dict}/nonsilence_lexicon.txt 

cat ${dict}/{,non}silence_lexicon.txt | LC_ALL=C sort -u > ${dict}/lexicon.txt

# Prepare the rest of the dictionary directory
# -----------------------------------------------
./local/prepare_dict.py \
  --silence-lexicon ${dict}/silence_lexicon.txt ${dict}/lexicon.txt ${dict}


