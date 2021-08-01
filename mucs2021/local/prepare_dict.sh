#!/bin/bash

# Copyright 2021 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

. ./path.sh

lexicon_type="baseline"
lang=hindi

. ./utils/parse_options.sh

mkdir -p data/dict

if [[ $lexicon_type = "baseline" ]]; then
  ./local/prepare_baseline_dict.sh --lang ${lang}
elif [[ $lexicon_type = "wikipron" ]]; then
  local/prepare_wikipron_dict.sh --lang ${lang}
fi  
