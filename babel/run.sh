#!/bin/bash

. ./path.sh
. ./cmd.sh

langid=multi
affix=
train_nj=120
stage=0
boost_sil=0.5

numLeavesTri1=1000
numGaussTri1=10000
numLeavesTri2=2500
numGaussTri2=36000
numLeavesTri3=2500
numGaussTri3=36000
numLeavesMLLT=2500
numGaussMLLT=360000
numLeavesSAT=2500
numGaussSAT=36000
numLeavesChain=3500

. ./utils/parse_options.sh

for l in `cat conf/train.list | awk '{print $1}'`; do
  langs+=("$l")    
done

# Data, dict, and lang prep
if [ $stage -le 0 ]; then
  echo "Preparing data"
  for l in ${langs[@]}; do 
    ./local/prepare_babel_data.sh --FLP true ${l}
  done
fi

if [ $stage -le 1 ]; then
  dicts_and_train=""
  for l in ${langs[@]}; do
    dicts_and_train="data/dict_${l} data/${l}_train ${dicts_and_train}" 
  done
  ./local/prepare_multilingual_data.sh \
    data/lang_${langid} data/${langid}_train exp/${langid} ${dicts_and_train}   
fi

trainset=${langid}_train
traindir=data/${trainset}
langdir=data/lang_${langid}${affix}
# Feature Prep
if [ $stage -le 2 ]; then
  ./steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj ${train_nj} \
    ${traindir} exp/make_mfcc_pitch/${trainset} mfcc
  ./utils/fix_data_dir.sh ${traindir}
  ./steps/compute_cmvn_stats.sh ${traindir} exp/make_mfcc_pitch/${trainset} mfcc
  ./utils/fix_data_dir.sh ${traindir}
  touch ${traindir}/.mfcc.done
fi

# Subset data for monophone trainin
if [ $stage -le 3 ]; then
  numutt=`cat ${traindir}/feats.scp | wc -l`
  if [ $numutt -gt 5000 ]; then
    local/subset_utts_by_lang.py ${traindir}/segments 5000 ${traindir}/sub1.list     
    utils/subset_data_dir.sh --utt-list ${traindir}/sub1.list ${traindir} ${traindir}_sub1
  else
    (cd data; ln -s ${trainset} ${trainset}_sub1)
  fi
  
  if [ $numutt -gt 10000 ] ; then
    local/subset_utts_by_lang.py ${traindir}/segments 10000 ${traindir}/sub2.list     
    utils/subset_data_dir.sh --utt-list ${traindir}/sub2.list ${traindir} ${traindir}_sub2
  else
    (cd data; ln -s ${trainset} ${trainset}_sub2 )
  fi
  
  if [ $numutt -gt 20000 ] ; then
    local/subset_utts_by_lang.py ${traindir}/segments 20000 ${traindir}/sub3.list     
    utils/subset_data_dir.sh --utt-list ${traindir}/sub3.list ${traindir} ${traindir}_sub3
  else
    (cd data; ln -s ${trainset} ${trainset}_sub3 )
  fi
  touch ${traindir}_sub3/.done
fi

###############################################################################
# HMM-GMM Training
############################################################################### 
if [ $stage -le 4 ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (small) monophone training in exp/mono on" `date`
  echo ---------------------------------------------------------------------
  steps/train_mono.sh \
    --boost-silence $boost_sil --nj 20 --cmd "$train_cmd" \
    ${traindir}_sub1 ${langdir} exp/mono
fi

if [ $stage -le 5 ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (small) triphone training in exp/tri1 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj ${train_nj} --cmd "$train_cmd" \
    ${traindir}_sub2 ${langdir} exp/mono exp/mono_ali_sub2

  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri1 $numGaussTri1 \
    ${traindir}_sub2 ${langdir} exp/mono_ali_sub2 exp/tri1

  touch exp/tri1/.done
fi

if [ $stage -le 6 ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (medium) triphone training in exp/tri2 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj ${train_nj} --cmd "$train_cmd" \
    ${traindir}_sub3 ${langdir} exp/tri1 exp/tri1_ali_sub3

  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 \
    ${traindir}_sub3 ${langdir} exp/tri1_ali_sub3 exp/tri2
  
  local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
    ${traindir}_sub3 ${langdir} exp/${langid}/lexicon \
    exp/tri2 exp/${langid}/dictp/tri2 exp/${langid}/langp/tri2 ${langdir}p/tri2
  
  touch exp/tri2/.done
fi

if [ $stage -le 7 ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (full) triphone training in exp/tri3 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    ${traindir} ${langdir}p/tri2 exp/tri2 exp/tri2_ali

  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesTri3 $numGaussTri3 ${traindir} ${langdir}p/tri2 exp/tri2_ali exp/tri3

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
    ${traindir} ${langdir} exp/${langid}/lexicon \
    exp/tri3 exp/${langid}/dictp/tri3 exp/${langid}/langp/tri3 ${langdir}p/tri3

  touch exp/tri3/.done
fi

if [ $stage -le 8 ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (lda_mllt) triphone training in exp/tri4 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    ${traindir} ${langdir}p/tri3 exp/tri3 exp/tri3_ali

  steps/train_lda_mllt.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT ${traindir} ${langdir}p/tri3 exp/tri3_ali exp/tri4

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
    ${traindir} ${langdir} exp/${langid}/lexicon \
    exp/tri4 exp/${langid}/dictp/tri4 exp/${langid}/langp/tri4 ${langdir}p/tri4

  touch exp/tri4/.done
fi

if [ $stage -le 9 ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (SAT) triphone training in exp/tri5 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    ${traindir} ${langdir}p/tri4 exp/tri4 exp/tri4_ali

  steps/train_sat.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT ${traindir} ${langdir}p/tri4 exp/tri4_ali exp/tri5

  local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
    ${traindir} ${langdir} exp/${langid}/lexicon \
    exp/tri5 exp/${langid}/dictp/tri5 exp/${langid}/langp/tri5 ${langdir}p/tri5

  touch exp/tri5/.done
fi

if [ $stage -le 10 ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/tri5_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_fmllr.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    ${traindir} ${langdir}p/tri5 exp/tri5 exp/tri5_ali
  
  local/reestimate_langp.sh --cmd "$train_cmd" --unk "<unk>" \
    ${traindir} ${langdir} exp/${langid}/lexicon \
    exp/tri5_ali exp/${langid}/dictp/tri5_ali exp/${langid}/langp/tri5_ali ${langdir}p/tri5_ali
   
  touch exp/tri5_ali/.done
fi
