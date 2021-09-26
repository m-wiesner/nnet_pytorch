#!/bin/bash

. ./path.sh
. ./cmd.sh

raw=false

. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: split_memmap_data.sh <datadir> <targets> <n>"
  exit 1;
fi

datadir=$1
targets=$2
num_split=$3

dataname=`basename ${datadir}`
mapped_dir=${datadir}/mapped # don't change this path
mkdir -p $mapped_dir
echo "$0: Splitting data in $num_split parts"

featname=feats
raw_opts=""
if $raw; then
  featname=wav
  raw_opts="--raw"
fi

# spread the mapped numpy arrays over various machines, as this data-set is quite large.
if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
  utils/create_split_dir.pl /export/b{11,12,13,14}/$USER/kaldi-data/egs/${dataname}_$(date +'%m_%d_%H_%M')/$mapped_dir/storage \
    $mapped_dir/storage
fi
utils/split_data.sh ${datadir} $num_split
for n in $(seq $num_split); do
  # the next command does nothing unless $mapped_feats_dir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $mapped_dir/${featname}.dat.$n
done
$train_cmd JOB=1:$num_split exp/make_fbank/${dataname}/memmap_data.JOB.log \
  memmap_data.py ${raw_opts} --utt-list ${targets} ${datadir}/split${num_split}/JOB/${featname}.scp $mapped_dir/feats.dat.JOB \
  $mapped_dir/metadata.JOB
echo $num_split > ${datadir}/num_split

