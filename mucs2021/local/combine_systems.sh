#!/bin/bash

. ./path.sh
. ./cmd.sh

# For now just manually specify the decode direcotries here.
decode_dirs="exp/wrn_sp_nopd/decode_185_215.mdl_graph_pd_expanded+extra_1.0_cw140_blindtest_rnnlm_0.4 \
             exp/blstm_librispeech/decode_120_160.mdl_graph_pd_expanded+extra_1.0_cw220_blindtest_rnnlm_0.4 \
             exp/blstm_librispeech_joint/decode_180_200.mdl_graph_pd_expanded+extra_1.0_cw220_blindtest_rnnlm_0.4"

. ./utils/parse_options.sh

if [ $# -le 3 ]; then
  echo "Usage: ./local/combine_systems.sh <data> <words> <odir>"
  exit 1;
fi

data=$1
words=$2
odir=$3

decode_dirs=( $decode_dirs )
num_sys=${#decode_dirs[@]}
mkdir -p ${odir}/{log,scoring}
for i in `seq 0 $[num_sys-1]`; do
  lats[$i]="\"ark:gunzip -c ${decode_dirs[$i]}/lat.*.gz | \""
done

$decode_cmd LMWT=10:10 $odir/log/combine_lats.LMWT.log \
  lattice-combine --inv-acoustic-scale=LMWT ${lats[@]} ark:- \| \
  lattice-mbr-decode --word-symbol-table=${words} ark:- \
  ark,t:${odir}/scoring/LMWT.tra || exit 1;

./local/score_latcomb.sh --cmd "$decode_cmd" --min-lmwt 10 --max-lmwt 10 \
  --apply-output-filter true \
  ${data} ${words} ${odir}

