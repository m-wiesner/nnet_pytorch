#!/bin/bash

. ./path.sh
. ./cmd.sh
. ./conf/lang.conf

stage=0
langs="104 201 307 404"
acoustic_mdl=exp/chain_wrn_3500
subsample=4
num_split=20
FLP=true
resolved=false
. ./utils/parse_options.sh

# Prepare the data, data splits, lexicons, and dictionaries for the
# test languages
if [ $stage -le 0 ]; then
  for l in ${langs}; do
    if [ -d data/${l}_train ]; then
      echo "Skipping ${l} because data/${l}_train already exists..."
      continue  
    fi
    ./local/prepare_babel_data.sh --FLP $FLP --make-dev true ${l}
    ./local/phoneset_diff.sh data/dict_${l}/lexicon.txt exp/multi/lexicon/lexicon.txt > data/dict_${l}/missing_phones.txt
  done
fi

# Check that there are no missing phones
if [ $stage -le 1 ]; then
  for l in ${langs}; do
    dict=data/dict_${l}
    if [ ! -f data/dict_${l}/missing_phones.txt ]; then
      echo "Expected file data/dict_${l}/missing_phones.txt to exist even if "
      echo "empty..."
      exit 1; 
    fi
    if [[ -s data/dict_${l}/missing_phones.txt && $resolved = "false" ]]; then
      echo "data/dict_${l}/missing_phones.txt was non-empty."
      echo "Resolve missing phonemes before continuing with stage 1."
      echo "   ./local/prepare_babel_test.sh --stage 1" 
      exit 1;
    fi
    
    cp -r ${dict} ${dict}_mapped
    cat ${dict}/lexicon.txt | ./utils/apply_map.pl -f 2- --permissive ${dict}/missing_phones.txt 2>/dev/null > ${dict}_mapped/lexicon.txt
    
    # Recreate the dictionary directory
    # extra_questions, nonsilence_phones.txt, silence_phones.txt etc..
    python local/prepare_dict.py \
      --silence-lexicon ${dict}/silence_lexicon.txt \
      ${dict}_mapped/lexicon.txt ${dict}_mapped   
   
    # Create the lang directory from the mapped dictionary directory
    ./utils/prepare_lang.sh \
      --phone-symbol-table data/lang_multip/tri5/phones.txt \
      --share-silence-phones true \
      ${dict}_mapped "<unk>" ${dict}_mapped/tmp.lang data/lang_${l} 
  
    # Train lm
    ./local/train_lm.sh data/lang_${l}/words.txt data/${l}_train/text data/lm_${l}
    # Create G.fst from arpa file
    ./utils/format_lm.sh data/lang_${l} data/lm_${l}/lm.gz data/dict_${l}_mapped/lexicon.txt data/lang_${l}
    ./utils/mkgraph.sh --self-loop-scale 1.0 \
      data/lang_${l} ${acoustic_mdl}/tree ${acoustic_mdl}/tree/graph_${l}

    ./utils/copy_data_dir.sh data/${l}_dev10h data/${l}_dev10h_fbank_64
    ./steps/make_fbank.sh --nj 100 data/${l}_dev10h_fbank_64
    ./steps/compute_cmvn_stats.sh data/${l}_dev10h_fbank_64
    ./utils/fix_data_dir.sh data/${l}_dev10h_fbank_64
    
    prepare_unlabeled_tgt.py --subsample ${subsample} \
      data/${l}_dev10h_fbank_64/utt2num_frames > data/${l}_dev10h_fbank_64/pdfid.${subsample}.tgt
    
    split_memmap_data.sh data/${l}_dev10h_fbank_64 data/${l}_dev10h_fbank_64/pdfid.${subsample}.tgt ${num_split}
  done
  
  echo "Successfully prepared data!" 
fi
