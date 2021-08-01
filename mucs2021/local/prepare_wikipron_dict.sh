#!/bin/bash

# Copyright 2021 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

. ./path.sh

lang=hindi

. ./utils/parse_options.sh

dict_dir=data/dict
g2p_dir=${dict_dir}/g2p

if [[ $lang = "hindi" ]]; then
  corpus_lex=https://raw.githubusercontent.com/navana-tech/baseline_recipe_is21s_indic_asr_challenge/master/is21-subtask2-kaldi/hindi_baseline/corpus/lang/lexicon.txt
  wikipron_lex=https://raw.githubusercontent.com/kylebgorman/wikipron/master/data/tsv/hin_deva_phonemic.tsv
elif [[ $lang = "bengali" ]]; then
  corpus_lex=https://raw.githubusercontent.com/navana-tech/baseline_recipe_is21s_indic_asr_challenge/master/is21-subtask2-kaldi/bengali_baseline/corpus/lang/lexicon.txt
  wikipron_lex=https://raw.githubusercontent.com/kylebgorman/wikipron/master/data/scrape/tsv/ben_beng_phonemic.tsv
fi

wget ${corpus_lex} -O ${dict_dir}/baseline.lex
wget ${wikipron_lex} -O ${dict_dir}/wikipron.lex

mkdir -p ${g2p_dir}
phonetisaurus-align --input=${dict_dir}/wikipron.lex --ofile=${g2p_dir}/g2p.corpus

ngram-count -lm ${g2p_dir}/g2p.arpa -maxent -maxent-convert-to-arpa \
  -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 2 \
  -kndiscount4 -gt4min 3 -order 4 -text ${g2p_dir}/g2p.corpus -sort 2>&1

phonetisaurus-arpa2wfst \
  --lm=${g2p_dir}/g2p.arpa \
  --ofile=${g2p_dir}/g2p.fst \
  --ssyms=${g2p_dir}/g2p.syms 2>&1

LC_ALL= python local/fix_lexicon.py --use-lang-tags --lang ${lang} \
  ${dict_dir}/{baseline,wikipron}.lex ${g2p_dir}/g2p.fst > ${dict_dir}/lexicon.unsorted.txt

LC_ALL=C sort ${dict_dir}/lexicon.unsorted.txt > ${dict_dir}/lexicon.txt

LC_ALL= python local/prepare_dict.py \
  --silence-lexicon <(echo -e "\!SIL SIL\n<unk> SPN") \
  ${dict_dir}/lexicon.txt ${dict_dir}
