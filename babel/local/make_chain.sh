#!/bin/bash
. ./path.sh
. ./cmd.sh

num_leaves=3500
lang_affix=_chain
subsampling=4
extra_text=
extra_lang=

. ./utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: ./local/make_chain.sh --num-leaves n <data> <lang> <ali> <chaindir>"
  exit 1;
fi

data=$1
lang=$2
alidir=$3
odir=$4

echo "Creating Chain Topology, Denominator Graph, and nnet Targets ..."
lang_chain=${lang}${lang_affix}_${num_leaves}
cp -r $lang $lang_chain
silphonelist=$(cat $lang_chain/phones/silence.csl) || exit 1;
nonsilphonelist=$(cat $lang_chain/phones/nonsilence.csl) || exit 1;

# Use our special topology... note that later on may have to tune this
# topology.
steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist > ${lang_chain}/topo

steps/nnet3/chain/build_tree.sh \
  --frame-subsampling-factor ${subsampling} \
  --context-opts "--context-width=2 --central-position=1" \
  --cmd "$train_cmd" ${num_leaves} ${data} \
  $lang_chain $alidir ${odir}/tree

extra_text_cmd="cat"
if [ ! -z $extra_text ]; then
  awk '(NR==FNR) {a[$1]=$0; next} !($1 in a){print $0}' ${data}/text $extra_text |\
  LC_ALL= python ./steps/chain/e2e/text_to_phones.py ${extra_lang} |\
  ./utils/sym2ing.pl -f 2- ${extra_lang}/phones.txt | ${odir}/extra_phones.txt 
  extra_text_cmd="cat ${odir}/extra_phones.txt -" 
fi

ali-to-phones ${odir}/tree/final.mdl ark:"gunzip -c ${odir}/tree/ali.*.gz |" ark:- |\
  ${extra_text_cmd} | chain-est-phone-lm --num-extra-lm-states=2000 ark:- ${odir}/phone_lm.fst

chain-make-den-fst ${odir}/tree/tree ${odir}/tree/final.mdl \
  ${odir}/phone_lm.fst ${odir}/den.fst ${odir}/normalization.fst
