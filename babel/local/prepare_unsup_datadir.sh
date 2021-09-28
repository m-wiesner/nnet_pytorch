#!/bin/bash

. ./path.sh
. ./cmd.sh
. ./conf/lang.conf

feat_affix=_fbank_64
lang=201
subsampling=4

. ./utils/parse_options.sh

unsup_whole=data/${lang}_unsup_whole
unsup_dir=data/${lang}_unsup
unsup_data_dir=unsup_data_dir_${lang}

unsup_sph_files=( `find -L ${!unsup_data_dir} -name "*.sph"` )
#mkdir -p ${unsup_whole}
#for f in ${unsup_sph_files[@]}; do
#  fname=`basename ${f%%.sph}`
#  uttid=$(echo ${fname} | awk -F'_' '{if ($7 == "inLine"){print $4"_A_"$5"_"$6} else{print $4"_B_"$5"_"$6}}')
#  echo "$uttid sph2pipe -f wav -p -c 1 ${f} |"
#done | sort > ${unsup_whole}/wav.scp
#awk '{print $1}' ${unsup_whole}/wav.scp | awk -F'_' '{print $1"_"$2"_"$3"_"$4" "$1"_"$2}' > ${unsup_whole}/utt2spk
#./utils/utt2spk_to_spk2utt.pl ${unsup_whole}/utt2spk > ${unsup_whole}/spk2utt 
#./steps/make_mfcc.sh --cmd "$train_cmd" --nj 32 ${unsup_whole}
#./utils/fix_data_dir.sh ${unsup_whole}
#./local/compute_vad_decision.sh ${unsup_whole}
#./local/vad_to_segments.sh --segmentation-opts "--silence-proportion 0.05 --max-segment-length 10" \
#  --cmd "$train_cmd" --nj 20 --min-duration 2.0 ${unsup_whole} ${unsup_dir} 
#awk '{sum+=$4-$3} END{print "speech (h): "sum/3600}' ${unsup_dir}/segments
#./utils/copy_data_dir.sh ${unsup_dir} ${unsup_dir}${feat_affix}
#./steps/make_fbank.sh --cmd "$train_cmd" --nj 32 ${unsup_dir}${feat_affix}
#./steps/compute_cmvn_stats.sh ${unsup_dir}${feat_affix}
#./utils/fix_data_dir.sh ${unsup_dir}${feat_affix}
prepare_unlabeled_tgt.py --subsample ${subsampling} \
  ${unsup_dir}${feat_affix}/utt2num_frames \
  > ${unsup_dir}${feat_affix}/pdfid.${subsampling}.tgt
split_memmap_data.sh ${unsup_dir}${feat_affix} ${unsup_dir}${feat_affix}/pdfid.${subsampling}.tgt 10

exit 0;

