#!/bin/bash

# Copyright 2021 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

map=local/hindi_english_all
affix="_norm"

. ./utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: ./local/normalize_datadir.sh <data> <dict> <odict> <olang>"
  exit 1;
fi

datadir=$1
dict=$2
odict=$3
olang=$4

./utils/copy_data_dir.sh ${datadir} ${datadir}${affix}
cat ${datadir}/text | ./utils/apply_map.pl -f 2- --permissive ${map} 2>/dev/null > ${datadir}${affix}/text

cp -r ${dict} ${odict}
rm ${odict}/lexicon*
LC_ALL= python local/normalize_lexicon.py ${dict}/lexicon.txt ${map} > ${odict}/lexion.raw
LC_ALL=C sort ${odict}/lexion.raw > ${odict}/lexicon.txt

./utils/prepare_lang.sh --share-silence-phones true \
  ${odict} "<unk>" ${odict}/tmp.lang $olang
 
 

