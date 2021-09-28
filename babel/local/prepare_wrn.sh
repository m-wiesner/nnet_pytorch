#!/bin/bash


. ./path.sh
. ./cmd.sh

data=data/multi_train
lang=data/lang_multip/tri5
ali=exp/tri5_ali
chaindir=exp/chain_wrn_3500
extra_text=
extra_lang=

. ./utils/parse_options.sh



# Make the chain directory. Supply external text and corresponding langdir
# using the --extra-text and --extra-lang flags
if [ -d $chaindir ]; then
  echo "Not preparing ${chaindir} as it already exists ..."
else
  ./local/make_chain.sh --extra-text ${extra_text} --extra-lang ${extra_lang} \
    ${data} ${lang} ${ali} ${chaindir}
fi

./utils/copy_data_dir.sh ${data} ${data}_fbank_64
./steps/make_fbank.sh --cmd "$train_cmd" --nj 200 ${data}_fbank_64
./steps/compute_cmvn_stats.sh ${data}_fbank_64

ali-to-pdf ${chaindir}/tree/final.mdl ark:"gunzip -c ${chaindir}/tree/ali.*.gz |" ark,t:${data}_fbank_64/pdfid.4.3500.tgt
split_memmap_data.sh --raw true data/multi_train_fbank_64 data/multi_train_fbank_64/pdfid.4.3500.tgt 80

