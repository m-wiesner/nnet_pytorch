#!/bin/bash

# Copyright 2021 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0


. ./path.sh
. ./cmd.sh

if [ $# -ne 5 ]; then
  echo "Usage: ./local/expand_lexicon.sh <data> <dict> <new_vocab> <lang> <wdir>"
  exit 1;  
fi

traindir=$1
dict=$2
vocab=$3
lang=$4
wdir=$5

mkdir -p ${wdir}/g2p/oov_words
mkdir -p ${dict}_expanded

comm -23 <(awk '{print $1}' ${vocab} | LC_ALL=C sort) \
         <(awk '{print $1}' ${dict}/lexicon.txt | LC_ALL=C sort -u) \
         > ${wdir}/oov_words

./steps/dict/train_g2p_phonetisaurus.sh \
  --silence-phones ${dict}/silence_phones.txt \
  --only-words true \
  ${dict}/lexicon.txt ${wdir}/g2p

phonetisaurus-g2pfst --model=${wdir}/g2p/model.fst \
  --wordlist=${wdir}/oov_words \
  --nbest=1 > ${wdir}/g2p/oov_words/lex_out

awk -F'\t' '{print $1,$3}' ${wdir}/g2p/oov_words/lex_out | awk '(NF > 2)' |\
  LC_ALL=C sort -u > ${wdir}/g2p/oov_words/lexicon_out

cat ${dict}/lexicon.txt ${wdir}/g2p/oov_words/lexicon_out |\
  LC_ALL=C sort -u > ${dict}_expanded/lexicon.txt

LC_ALL= python local/prepare_dict.py \
  --silence-lexicon <(echo -e "\!SIL SIL\n<unk> SPN") \
  ${dict}_expanded/lexicon.txt ${dict}_expanded

./utils/prepare_lang.sh --phone-symbol-table ${lang}/phones.txt \
  --share-silence-phones true \
  ${dict}_expanded "<unk>" ${dict}_expanded/tmp.lang ${lang}_expanded

./local/train_lm.sh ${dict}_extended/lexicon.txt ${traindir}/text ${lang}_expanded
./utils/format_lm.sh ${lang}_expanded ${lmdir}_expanded/lm.gz ${dict}_expanded/lexicon.txt ${lang}

