#!/bin/bash

# Copyright 2021 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

root=/export/corpora5/MultilingualCodeSwitching2021/codeswitching/104

. ./path.sh
. ./cmd.sh

stage=0
stop_stage=8
lexicon_type="wikipron" # baseline wikipron
normalize=true
librispeech_model= # path to pretrained librispeech model goes here
lang=hindi

. ./utils/parse_options.sh

if [[ $lang = "hindi" ]]; then
  data=${root}/Hindi-English 
elif [[ $lang = "bengali" ]]; then
  data=${root}/Bengal-English
else
  echo "No data exists for language ${lang}"
  exit 1;
fi

if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
  ./local/prepare_data.sh ${data}
  ./local/prepare_dict.sh --lexicon-type ${lexicon_type} --lang ${lang}
  ./utils/prepare_lang.sh --share-silence-phones true \
    data/dict "<unk>" data/dict/tmp.lang data/lang_${lexicon_type}_nosp 
fi

# Feature extraction
# Stage 2: MFCC feature extraction + mean-variance normalization
if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
   for x in train test; do
      steps/make_mfcc.sh --nj 40 --cmd "$train_cmd" data/$x exp/make_mfcc/$x mfcc
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc
   done
fi

if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  utils/subset_data_dir.sh --shortest data/train 500 data/train_500short
  utils/subset_data_dir.sh data/train 5000 data/train_5k
  utils/subset_data_dir.sh data/train 10000 data/train_10k
fi

# Stage 3: Training and decoding monophone acoustic models
if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  ### Monophone
  echo "mono training"
  steps/train_mono.sh --boost-silence 1.25 --nj 30 --cmd "$train_cmd" data/train_500short data/lang_${lexicon_type}_nosp exp/mono
fi

# Stage 4: Training tied-state triphone acoustic models
if [ $stage -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  ### Triphone
  echo "tri1 training"
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
    data/train_5k data/lang_${lexicon_type}_nosp exp/mono exp/mono_ali_train_5k
  steps/train_deltas.sh --boost-silence 1.25  --cmd "$train_cmd"  \
    2000 10000 data/train_5k data/lang_${lexicon_type}_nosp exp/mono_ali_train_5k exp/tri1
    echo "tri1 training done"
fi

# More data
if [ $stage -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "tri2b training"
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
    data/train_10k data/lang_${lexicon_type}_nosp exp/tri1 exp/tri1_ali_train_10k

  steps/train_deltas.sh --boost-silence 1.25  --cmd "$train_cmd"  \
    2500 15000 data/train_10k data/lang_${lexicon_type}_nosp exp/tri1_ali_train_10k exp/tri1b
fi

# Stage 5: Train an LDA+MLLT system.
if [ $stage -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
      data/train data/lang_${lexicon_type}_nosp exp/tri1b exp/tri1b_ali
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 4200 40000 \
    data/train_10k data/lang_${lexicon_type}_nosp exp/tri1b_ali exp/tri2
fi

# Stage 6: Train tri3b, which is LDA+MLLT+SAT
if [ $stage -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  steps/align_si.sh --use-graphs true --nj 40 --cmd "$train_cmd" \
    data/train data/lang_${lexicon_type}_nosp exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
    data/train data/lang_${lexicon_type}_nosp exp/tri2_ali exp/tri3
fi

# Add silence probs
if [ $stage -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  datadir=data/train
  dict=data/dict
  lang=data/lang_${lexicon_type}
  alidir=exp/tri3
  if $normalize; then
    ./local/normalize_datadir.sh --affix "_norm" --map local/${lang}_english_all \
      data/train data/dict data/dict_norm data/lang_${lexicon_type}_nosp_norm 
    datadir=data/train_norm
    dict=data/dict_norm
    lang=data/lang_${lexicon_type}_norm
    alidir=exp/tri3_ali_norm
    steps/align_fmllr.sh --cmd "$train_cmd" --nj 40 \
      ${datadir} data/lang_${lexicon_type}_nosp_norm exp/tri3 ${alidir}
  fi

  steps/get_prons.sh --cmd "$train_cmd" \
    ${datadir} data/lang_${lexicon_type}_nosp_norm ${alidir}
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    ${dict} \
    ${alidir}/pron_counts_nowb.txt ${alidir}/sil_counts_nowb.txt \
    ${alidir}/pron_bigram_counts_nowb.txt ${dict}_sp

  utils/prepare_lang.sh --share-silence-phones true ${dict}_sp \
    "<unk>" ${dict}_sp/.lang_tmp ${lang}

  ./steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
    ${datadir} ${lang} exp/tri3 exp/tri3_ali
fi

datadir=data/train
dict=data/dict
lang=data/lang_${lexicon_type}
alidir=exp/tri3_ali
lmdir=data/lm
graph=graph
if $normalize; then
  datadir=data/train_norm
  dict=data/dict_norm
  lang=data/lang_${lexicon_type}_norm
  alidir=exp/tri3_ali_norm
  lmdir=data/lm_norm
  graph=graph_norm
fi

# Data cleanup for nnet_pytorch or Kaldi training 
if [ $stage -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  ./steps/cleanup/clean_and_segment_data.sh --cmd "$train_cmd" --nj 80 ${datadir} ${lang} exp/tri3 exp/cleanup ${datadir}_cleaned
fi 

# make LM and decode GMM system
if [ $stage -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  num_utts=`cat data/train/text | wc -l`
  num_valid_utts=$(($num_utts/10))
  num_train_utts=$(($num_utts - $num_valid_utts)) 
  
  mkdir -p ${lmdir}
  shuf ${datadir}/text > ${lmdir}/text.shuf
  
  ./local/train_lm.sh ${dict}_sp/lexicon.txt ${lmdir}/text.shuf ${lmdir}
  ./utils/format_lm.sh ${lang} ${lmdir}/lm.gz ${dict}_sp/lexicon.txt ${lang}

  ./utils/mkgraph.sh ${lang} exp/tri3 exp/tri3/${graph}

  ./steps/decode_fmllr_extra.sh --cmd "$decode_cmd" --nj 30 exp/tri3/${graph} data/test exp/tri3/decode_${graph}_test
fi

# Pronunciation Learning from Acoustics
if [ $stage -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  ./steps/dict/learn_lexicon_greedy.sh --retrain-src-mdl false \
    --oov-symbol "<unk>" --nj 80 --cmd "$train_cmd" \
    ${dict} ${lmdir}/vocab ${datadir} exp/tri3 ${lang} ${dict}_pd exp/debug_lexicon

  ./utils/prepare_lang.sh --share-silence-phones true ${dict}_pd "<unk>" ${dict}_pd/.lang_tmp ${lang}_pd_nosp
  ./steps/align_fmllr.sh --cmd "$train_cmd" --nj 60 ${datadir} ${lang}_pd_nosp exp/tri3 exp/tri3_ali_pd_train
  ./steps/get_prons.sh --cmd "$train_cmd" ${datadir} data/lang_wikipron_pd_nosp exp/tri3_ali_pd_train
  ./utils/dict_dir_add_pronprobs.sh --max-normalize true \
    ${dict}_pd exp/tri3_ali_pd_train/pron_counts_nowb.txt \
    exp/tri3_ali_pd_train/sil_counts_nowb.txt \
    exp/tri3_ali_pd_train/pron_bigram_counts_nowb.txt data/dict_pd_sp

  ./utils/prepare_lang.sh --share-silence-phones true \
    ${dict}_pd_sp "<unk>" ${dict}_pd_sp/.lang_tmp ${lang}_pd
fi

# Recreate lang directory and dict with expanded vocabulary (for decoding)
if [ $stage -le 12 ] && [ ${stop_stage} -ge 12 ]; then
  ./local/expand_lexicon.sh ${datadir} ${dict}_pd_sp ${vocab} ${lang}_pd exp/expand_lexicon 
fi

# Still need to add LM scripts. For now you can download the langdir we used
# or you can use the default lm which still gives decent results.
# if [ $stage -le 13 ] && [ ${stop_stage} -ge 13 ]; then
#  ./local/prepare_lm.sh
# fi
#
#

echo "Finished GMM-HMM training and creating decoding graphs. See run.sh at the"
"     bottom of the file for nnet_pytorch training options."
exit

#                NNET_PYTORCH TRAINING & DECODING
# To run nnet_pytorch make sure path.sh is setup approrpiately.


################################# Training ####################################
# To run nnet_pytorch we first create the speed and volume perturbed features
# and alignments. From this we make the "chain" diretory which has the
# denominator graph as well as phone_lm.fst needed for lf-MMI training 
./local/nnet_pytorch/prepare_train.sh --ali-affix _cleaned_sp \
                                      --feat-affix _fbank_64 \
                                      data/train_norm_cleaned data/lang_wikipron_norm exp/tri3

# The chain directory only needs to be created once and can be shared across
# all systems.
./local/nnet_pytorch/make_chain.sh data/train_norm_cleaned_sp_fbank_64 \
  data/lang_wikipron_norm exp/tri3_ali_norm_cleaned_sp exp/chain_cleaned_sp

./local/nnet_pytorch/run-wrn.sh
./local/nnet_pytorch/run-blstm.sh

# To run the blstm initialized from librispeech blstm
if [ ! -z ${librispeech_model} ] && [ -f ${librispeech_model} ]; then
  ./local/nnet_pytorch/run-blstm.sh --init ${librispeech_model}
fi

# Assumes an existing chain directory and targets for bengali exits. Run the
# recipe with lang=bengali to get bengali alignments and chain directory 
# before running this part.
./local/nnet_pytoch/run-blstm-multi.sh --init ${librispeech_model}




#################### Decoding #################################################
# To decode the test data with an existing model use for instance
./local/prepare_blind_test.sh
./local/nnet_pytorch/prepare_test.sh data/blindtest

# Decode BLSTM Initialized with librispeech
./local/nnet_pytorch/decode.sh --testsets "blindtest" --checkpoint 120_160.mdl --model-dirname blstm_librispeech

# Decode WRN
./local/nnet_pytorch/decode.sh --testsets "blindtest" --checkpoint 185_215.mdl --model-dirname wrn_sp_nopd

# Decode BLSTM trained multilingually on bengali and hindi alignments
# To decode from the bengali side, use the --output-idx 1 flag

# hindi decoding from multilingual model
./local/nnet_pytorch/decode.sh --testsets "blindtest" --checkpoint 180_200.mdl --model-dirname blstm_librispeech_joint

# bengali decoding from multilingual model
if [[ ${lang} = "bengali" ]]; then
  ./local/nnet_pytorch/decode.sh --output-idx 1 --testsets "blindtest" --checkpoint 180_200.mdl --model-dirname blstm_librispeech_joint
fi

