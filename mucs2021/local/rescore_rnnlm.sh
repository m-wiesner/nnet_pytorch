#!/bin/bash

. ./path.sh
. ./cmd.sh

rnnlm_weight=0.4
ngram_order=4

. ./utils/parse_options.sh
if [ $# -ne 4 ]; then
  echo "Usage: ./local/rescore_rnnlm.sh <data> <lang> <rnnlm> <decode>"
  exit 1;
fi

data=$1
lang=$2
rnnlm=$3
decode_dir=$4

rnnlm/lmrescore_pruned.sh \
  --cmd "$decode_cmd" --mem 4G\
  --weight ${rnnlm_weight} \
  --max-ngram-order $ngram_order \
  ${lang} ${rnnlm} ${data} ${decode_dir} \
  ${decode_dir}_rnnlm_${rnnlm_weight}
