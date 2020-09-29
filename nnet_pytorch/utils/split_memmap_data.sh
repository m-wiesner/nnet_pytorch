#!/bin/bash

. ./path.sh
. ./cmd.sh

. ./utils/parse_options.sh
if [ $# -ne 2 ]; then
  echo "Usage: ./local/split_memmap_data.sh <datadir> <n>"
  exit 1;
fi

datadir=$1
num_split=$2

dataname=`basename ${datadir}`
mapped_dir=${datadir}/mapped # don't change this path
mkdir -p $mapped_dir
echo "$0: Splitting data in $num_split parts"
# spread the mapped numpy arrays over various machines, as this data-set is quite large.
if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
  utils/create_split_dir.pl /export/b{11,12,13,14}/$USER/kaldi-data/egs/librispeech100/$mapped_dir/storage \
    $mapped_dir/storage
fi
utils/split_data.sh ${datadir} $num_split
for n in $(seq $num_split); do
  # the next command does nothing unless $mapped_feats_dir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $mapped_dir/feats.dat.$n
done
$train_cmd JOB=1:$num_split exp/make_fbank/${dataname}/memmap_data.JOB.log \
  memmap_data.py ${datadir}/split${num_split}/JOB/feats.scp $mapped_dir/feats.dat.JOB \
  $mapped_dir/metadata.JOB
echo $num_split > ${datadir}/num_split

