#!/bin/bash

# Copyright 2021 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

order=4
oov_symbol="<unk>"
cmd=run.pl
stage=1
outlm=lm.gz

. ./utils/parse_options.sh

if [ $# -eq 0 ]; then
  echo "Usage: ./local/lm_train_dev_splits.sh <dict_words> <train> <dev> <olmdir>"
  exit 1
fi

dict_words=$1
train=$2
odir=$3

[ ! -d $odir ] && mkdir -p $odir

awk '{print $1}' $dict_words | grep -v '<eps>' | grep -v '\#0' | grep -v -F "$oov_symbol" > ${odir}/vocab
cut -d' ' -f2- $train > ${odir}/train


if [ $stage -le 2 ]; then
  echo "-------------------"
  echo "Maxent 4grams"
  echo "-------------------"
  sed 's/'${oov_symbol}'/<unk>/g' ${odir}/train | \
    ngram-count -lm - -order $order -text - -vocab ${odir}/vocab -unk -sort -maxent -maxent-convert-to-arpa|\
    sed 's/<unk>/'${oov_symbol}'/g' | gzip -c > ${odir}/lm.gz || exit 1
fi

