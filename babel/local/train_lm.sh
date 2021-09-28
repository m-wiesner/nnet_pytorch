#!/bin/bash

oov_symbol="<unk>"
cmd=run.pl
stage=1
outlm=lm.gz

. ./utils/parse_options.sh

if [ $# -eq 0 ]; then
  echo "Usage: ./local/lm_train_dev_splits.sh <dict_words> <train> <dev> <olmdir>"
  exit 1
fi

if [ $# -eq 4 ]; then
  dict_words=$1
  train=$2
  dev=$3
  odir=$4
elif [ $# -eq 3 ]; then
  dict_words=$1
  train=$2
  odir=$3
  dev=
fi

[ ! -d $odir ] && mkdir -p $odir

awk '{print $1}' $dict_words | grep -v '<eps>' | grep -v '\#0' | grep -v -F "$oov_symbol" > ${odir}/vocab
if [ ! -z $dev ]; then
  cut -d' ' -f2- $train > ${odir}/train
  cut -d' ' -f2- $dev > ${odir}/dev 
else
  num_utts=`cat $train | wc -l`
  num_dev=$((num_utts / 10))
  num_train=$((num_utts - num_dev))
  shuf $train > ${odir}/text
  head -n ${num_train} ${odir}/text | cut -d' ' -f2- > ${odir}/train
  tail -n ${num_dev} ${odir}/text | cut -d' ' -f2- > ${odir}/dev 
fi

if [ $stage -le 1 ]; then
  echo "-----------------------------"
  echo "Training LM"
  echo "_____________________________"
  # Compute LM
  echo "-------------------"
  echo "Good-Turing 2grams:"
  echo "-------------------"
  ngram-count -lm ${odir}/2gram.gt01.gz -gt1min 0 -gt2min 1 -order 2 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/2gram.gt02.gz -gt1min 0 -gt2min 2 -order 2 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  
  echo "-------------------"
  echo "Kneser-Ney 2grams:"
  echo "-------------------"
  ngram-count -lm ${odir}/2gram.kn01.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -order 2 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/2gram.kn02.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -order 2 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  
  echo "-------------------"
  echo "Good-Turing 3grams:"
  echo "-------------------"
  ngram-count -lm ${odir}/3gram.gt011.gz -gt1min 0 -gt2min 1 -gt3min 1 -order 3 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/3gram.gt012.gz -gt1min 0 -gt2min 1 -gt3min 2 -order 3 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/3gram.gt022.gz -gt1min 0 -gt2min 2 -gt3min 2 -order 3 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/3gram.gt023.gz -gt1min 0 -gt2min 2 -gt3min 3 -order 3 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  
  
  echo "-------------------"
  echo "Kneser-Ney 3grams:"
  echo "-------------------"
  ngram-count -lm ${odir}/3gram.kn011.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 1 -order 3 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/3gram.kn012.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 2 -order 3 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/3gram.kn022.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 2 -order 3 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/3gram.kn023.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 3 -order 3 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  
  
  
  echo "-------------------"
  echo "Good-Turing 4grams:"
  echo "-------------------"
  ngram-count -lm ${odir}/4gram.gt0111.gz -gt1min 0 -gt2min 1 -gt3min 1 -gt4min 1 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/4gram.gt0112.gz -gt1min 0 -gt2min 1 -gt3min 1 -gt4min 2 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/4gram.gt0122.gz -gt1min 0 -gt2min 1 -gt3min 2 -gt4min 2 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/4gram.gt0123.gz -gt1min 0 -gt2min 1 -gt3min 2 -gt4min 3 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/4gram.gt0113.gz -gt1min 0 -gt2min 1 -gt3min 1 -gt4min 3 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/4gram.gt0222.gz -gt1min 0 -gt2min 2 -gt3min 2 -gt4min 2 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/4gram.gt0223.gz -gt1min 0 -gt2min 2 -gt3min 2 -gt4min 3 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  
  
  echo "-------------------"
  echo "Kneser-Ney 4grams:"
  echo "-------------------"
  ngram-count -lm ${odir}/4gram.kn0111.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 1 -kndiscount4 -gt4min 1 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/4gram.kn0112.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 1 -kndiscount4 -gt4min 2 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/4gram.kn0113.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 1 -kndiscount4 -gt4min 3 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/4gram.kn0122.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 2 -kndiscount4 -gt4min 2 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/4gram.kn0123.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 1 -kndiscount3 -gt3min 2 -kndiscount4 -gt4min 3 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/4gram.kn0222.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 2 -kndiscount4 -gt4min 2 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
  ngram-count -lm ${odir}/4gram.kn0223.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 2 -kndiscount4 -gt4min 3 -order 4 -text ${odir}/train -vocab ${odir}/vocab -unk -sort -map-unk "$oov_symbol"
fi
  
  
if [ $stage -le 2 ]; then
  #please not that if the switch -map-unk "$oov_symbol" is used with -maxent-convert-to-arpa, ngram-count will segfault
  #instead of that, we simply output the model in the maxent format and convert it using the "ngram"
  echo "-----------------------------"
  echo "Training Maxent LM"
  echo "_____________________________"
  # Compute LM
  echo "-------------------"
  echo "Maxent 2grams"
  echo "-------------------"
  sed 's/'${oov_symbol}'/<unk>/g' ${odir}/train | \
    ngram-count -lm - -order 2 -text - -vocab ${odir}/vocab -unk -sort -maxent -maxent-convert-to-arpa|\
    sed 's/<unk>/'${oov_symbol}'/g' | gzip -c > ${odir}/2gram.me.gz || exit 1

  echo "-------------------"
  echo "Maxent 3grams"
  echo "-------------------"
  sed 's/'${oov_symbol}'/<unk>/g' ${odir}/train | \
    ngram-count -lm - -order 3 -text - -vocab ${odir}/vocab -unk -sort -maxent -maxent-convert-to-arpa|\
    sed 's/<unk>/'${oov_symbol}'/g' | gzip -c > ${odir}/3gram.me.gz || exit 1

  echo "-------------------"
  echo "Maxent 4grams"
  echo "-------------------"
  sed 's/'${oov_symbol}'/<unk>/g' ${odir}/train | \
    ngram-count -lm - -order 4 -text - -vocab ${odir}/vocab -unk -sort -maxent -maxent-convert-to-arpa|\
    sed 's/<unk>/'${oov_symbol}'/g' | gzip -c > ${odir}/4gram.me.gz || exit 1
fi

if [ $stage -le 3 ]; then
  echo "--------------------"
  echo "Computing perplexity"
  echo "--------------------"
  (
    for f in ${odir}/3gram* ; do ( echo $f; ngram -order 3 -lm $f -unk -map-unk "$oov_symbol" -ppl ${odir}/dev ) | paste -s -d ' ' ; done
    for f in ${odir}/4gram* ; do ( echo $f; ngram -order 4 -lm $f -unk -map-unk "$oov_symbol" -ppl ${odir}/dev ) | paste -s -d ' ' ; done
  )  | sort  -r -n -k 15,15g | column -t | tee ${odir}/perplexities.txt
fi

if [ $stage -le 4 ]; then
  nof_trigram_lm=`head -n 2 ${odir}/perplexities.txt | grep 3gram | wc -l`
  if [[ $nof_trigram_lm -eq 0 ]] ; then
    lmfilename=`head -n 1 ${odir}/perplexities.txt | cut -f 1 -d ' '`
  elif [[ $nof_trigram_lm -eq 2 ]] ; then
    lmfilename=`head -n 1 ${odir}/perplexities.txt | cut -f 1 -d ' '`
  else  #exactly one 3gram LM
    lmfilename=`head -n 2 ${odir}/perplexities.txt | grep 3gram | cut -f 1 -d ' '`
  fi
  (cd ${odir}; ln -sf `basename $lmfilename` $outlm )
fi

 
