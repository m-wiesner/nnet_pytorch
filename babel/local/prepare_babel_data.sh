#!/bin/bash

. ./conf/lang.conf
. ./path.sh
. ./cmd.sh

FLP=true
make_dev=false
skip_train=false

. ./utils/parse_options.sh
if [ $# -ne 1 ]; then
  echo >&2 "Usage: ./local/prepare_babel_data.sh <lang_id>" 
  exit 1;
fi
l=$1

l_suffix=${l}
if $FLP; then
  l_suffix=${l_suffix}_FLP
fi

mkdir -p data/local_${l}
mkdir -p data/dict_${l}
lexicon_file=lexicon_file_${l_suffix}
lexiconFlags=lexiconFlags_${l_suffix}
lexicon=data/local_${l}/lexicon.txt
train_data_dir=train_data_dir_${l_suffix}
train_data_list=train_data_list_${l_suffix}

echo "Lexicon: $lexicon_file"
echo "lexiconFlags: $lexiconFlags"
echo "lexicon: $lexicon"
echo "dir: ${!train_data_dir}"
echo "list: ${!train_data_list}"

if ! $skip_train; then
  local/make_corpus_subset.sh "${!train_data_dir}" "${!train_data_list}" ./data/raw_${l}
  train_data_dir=`utils/make_absolute.sh ./data/raw_${l}`
  local/make_lexicon_subset.sh $train_data_dir/transcription ${!lexicon_file} data/local_${l}/filtered_lexicon.txt
  cut -f1 data/local_${l}/filtered_lexicon.txt > data/local_${l}/words.txt 
  
  ./local/prepare_universal_dict.sh --istest ${make_dev} --dict data/dict_${l} --src data/local_${l} ${l}
  
  mkdir -p data/${l}_train
  local/prepare_acoustic_training_data.pl \
    --vocab data/dict_${l}/lexicon.txt --fragmentMarkers \-\*\~ \
    $train_data_dir data/${l}_train > data/${l}_train/skipped_utts.log
fi

if $make_dev; then
  dev10h_data_dir=dev10h_data_dir_${l}
  dev10h_data_list=dev10h_data_list_${l}
  local/make_corpus_subset.sh "${!dev10h_data_dir}" "${!dev10h_data_list}" ./data/raw_${l}_dev10h
  dev10h_data_dir=`utils/make_absolute.sh ./data/raw_${l}_dev10h`
  
  mkdir -p data/${l}_dev10h
  local/prepare_acoustic_training_data.pl --fragmentMarkers \-\*\~ \
    $dev10h_data_dir data/${l}_dev10h > data/${l}_dev10h/skipped_utts.log       

  ./local/prepare_stm.pl --fragmentMarkers \-\*\~ data/${l}_dev10h
fi
