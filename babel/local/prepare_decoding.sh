#!/bin/bash

dict_orig=
lang_orig=
prepare_training_data=false
decode_lang=data/lang_test
chaindir=exp/chain_wrn_3500
graphname=graph

stage=0
resolved=false
dict=exp/multi/lexicon
training_text=data/multi_train/text

. ./utils/parse_options.sh

if [ $stage -le 0 ]; then
  if $prepare_training_data; then
    # Assumes lang id is first row of conf/train.list
    langid=$(cat conf/train.list | head -1)
    ./local/prepare_babel_data.sh --skip-train false --make-dev false ${langid}
    ./local/prepare_multilingual_data.sh data/lang_multi data/multi_train \
      exp/multi data/dict_${langid} data/${langid}_train 
    dict=exp/multi/lexicon
    training_text=data/multi_train/text
  fi
  
  if [ ! -z $dict_orig ]; then
    ./local/phoneset_diff.sh ${dict}/lexicon.txt $dict_orig/lexicon.txt > ${dict}/missing_phones.txt
  else
    touch ${dict}/missing_phones.txt
  fi
fi 

# Check that there are no missing phones
if [ $stage -le 1 ]; then
  if [ ! -f ${dict}/missing_phones.txt ]; then
    echo "Expected file ${dict}/missing_phones.txt to exist even if "
    echo "empty..."
    exit 1; 
  fi
  if [[ -s ${dict}/missing_phones.txt && $resolved == "false" ]]; then
    echo "${dict}/missing_phones.txt was non-empty."
    echo "Resolve missing phonemes before continuing with stage 1."
    echo "   ./local/prepare_decoding.sh --stage 1" 
    exit 1;
  fi
  cp -r ${dict} ${dict}_mapped
  cat ${dict}/lexicon.txt |\
    ./utils/apply_map.pl -f 2- --permissive ${dict}/missing_phones.txt 2>/dev/null > ${dict}_mapped/lexicon.txt

  python local/prepare_dict.py \
    --silence-lexicon ${dict}/silence_lexicon.txt \
    ${dict}_mapped/lexicon.txt ${dict}_mapped
 
  if [ ! -z $lang_orig ]; then 
    ./utils/prepare_lang.sh \
      --phone-symbol-table ${lang_orig}/phones.txt \
      --share-silence-phones true \
      ${dict}_mapped "<unk>" ${dict}_mapped/tmp.lang ${decode_lang}
  else
    ./utils/prepare_lang.sh \
      --share-silence-phones true \
      ${dict}_mapped "<unk>" ${dict}_mapped/tmp.lang ${decode_lang}
  fi

  ./local/train_lm.sh ${decode_lang}/words.txt $training_text data/lm
  ./utils/format_lm.sh ${decode_lang} data/lm/lm.gz ${dict}_mapped/lexicon.txt ${decode_lang}
  ./utils/mkgraph.sh --self-loop-scale 1.0 \
    ${decode_lang} ${chaindir}/tree ${chaindir}/tree/${graphname}
  awk '{print $2,$1}' ${dict}/wordmap.0 > ${dict}/mapword.0
fi 
