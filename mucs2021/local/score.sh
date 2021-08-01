#!/usr/bin/env bash
# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey, Yenda Trmal)
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
stage=0
stats=true
beam=6
word_ins_penalty=0.0,0.5,1.0
min_lmwt=7
max_lmwt=17
#end configuration section.

echo "$0 $@"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1
lang_or_graph=$2
dir=$3

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done

ref_filtering_cmd="cat"
hyp_filtering_cmd="cat"
[ -x local/wer_output_filter ] && ref_filtering_cmd="./local/wer_output_filter"
[ -x local/wer_output_filter ] && hyp_filtering_cmd="./local/wer_output_filter"
# Assume that the last _***** is the only part that specifies the utterance
# and before that specifies the recording

scoringdir=$dir/scoring
mkdir -p $scoringdir

if [ $stage -le 1 ]; then
  if [ -f $data/text ]; then
    cat $data/text | ./local/combine_line_txt_to_paragraph.py \
      > $scoringdir/test.txt
    cat $scoringdir/test.txt | $ref_filtering_cmd > $scoringdir/test.filt.txt
  fi

  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    mkdir -p ${scoringdir}/penalty_$wip
    $cmd LMWT=$min_lmwt:$max_lmwt $scoringdir/penalty_$wip/log/best_path.LMWT.log \
      lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
      lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
      lattice-best-path --word-symbol-table=$symtab ark:- ark,t:- \| \
      utils/int2sym.pl -f 2- $symtab '>' $scoringdir/penalty_$wip/LMWT.txt || exit 1;
  
    for lmwt in $(seq $min_lmwt $max_lmwt); do
      cat $scoringdir/penalty_$wip/${lmwt}.txt | \
      ./local/combine_line_txt_to_paragraph.py \
      > $scoringdir/penalty_$wip/${lmwt}.combine.txt
  
      cat $scoringdir/penalty_$wip/${lmwt}.combine.txt |\
        ${hyp_filtering_cmd} > $scoringdir/penalty_$wip/${lmwt}.combine.filt.txt
    done
  done
fi

if [ $stage -le 2 ]; then
  if [ ! -f ${data}/text ]; then
    echo "Not scoring since no reference found."
    exit 0;
  fi
  
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    for ext in .txt .filt.txt; do
      if [[ $ext = ".txt" ]]; then
        wer_affix="_"
      else
        wer_affix="_filt_"
      fi
      $cmd LMWT=$min_lmwt:$max_lmwt $scoringdir/penalty_$wip/log/score.LMWT${ext}.log \
        cat $scoringdir/penalty_$wip/LMWT.combine${ext} \| \
        compute-wer --mode=present \
        ark:$scoringdir/test${ext}  ark,p:- ">&" $dir/wer${wer_affix}LMWT_$wip || exit 1;
    
      if [ -f ${data}/convs_dup ]; then
        mkdir -p ${scoringdir}_{,no}dup/penalty_$wip/log
        for lmwt in $(seq $min_lmwt $max_lmwt); do
          lmwt_file=$scoringdir/penalty_$wip/${lmwt}.combine${ext}
          awk '(NR==FNR){a[$1]=1; next} ($1 in a){print $0}' ${data}/convs_dup ${lmwt_file} \
            > ${scoringdir}_dup/penalty_$wip/${lmwt}.combine${ext}
          awk '(NR==FNR){a[$1]=1; next} ($1 in a){print $0}' ${data}/convs_nodup ${lmwt_file} \
            > ${scoringdir}_nodup/penalty_$wip/${lmwt}.combine${ext}
        done    
     
        $cmd LMWT=$min_lmwt:$max_lmwt ${scoringdir}_dup/penalty_$wip/log/score.LMWT${ext}.log \
          cat ${scoringdir}_dup/penalty_$wip/LMWT.combine${ext} \| \
          compute-wer --mode=present \
          ark:${scoringdir}/test${ext}  ark,p:- ">&" $dir/wer_dup${wer_affix}LMWT_$wip || exit 1;
        $cmd LMWT=$min_lmwt:$max_lmwt ${scoringdir}_dup/penalty_$wip/log/score.LMWT${ext}.log \
          cat ${scoringdir}_nodup/penalty_$wip/LMWT.combine${ext} \| \
          compute-wer --mode=present \
          ark:${scoringdir}/test${ext}  ark,p:- ">&" $dir/wer_nodup${wer_affix}LMWT_$wip || exit 1;
      fi
    done
  done
  
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    for lmwt in $(seq $min_lmwt $max_lmwt); do
      # adding /dev/null to the command list below forces grep to output the filename
      grep WER $dir/wer_${lmwt}_${wip} /dev/null
    done
  done | utils/best_wer.sh  >& ${scoringdir}/best_wer || exit 1
fi

best_wer_file=$(awk '{print $NF}' ${scoringdir}/best_wer)
best_wip=$(echo $best_wer_file | awk -F_ '{print $NF}')
best_lmwt=$(echo $best_wer_file | awk -F_ '{N=NF-1; print $N}')

if [ -z "$best_lmwt" ]; then
  echo "$0: we could not get the details of the best WER from the file $dir/wer_*.  Probably something went wrong."
  exit 1;
fi

if $stats; then
  mkdir -p ${scoringdir}/wer_details
  echo $best_lmwt > ${scoringdir}/wer_details/lmwt # record best language model weight
  echo $best_wip > ${scoringdir}/wer_details/wip # record best word insertion penalty
  
  $cmd ${scoringdir}/log/stats1.log \
    cat ${scoringdir}/penalty_$best_wip/$best_lmwt.combine.filt.txt \| \
    align-text --special-symbol="'***'" ark:${scoringdir}/test.filt.txt ark:- ark,t:- \|  \
    utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \| tee ${scoringdir}/wer_details/per_utt \|\
     utils/scoring/wer_per_spk_details.pl $data/utt2spk \> ${scoringdir}/wer_details/per_spk || exit 1;
  
  $cmd ${scoringdir}/log/stats2.log \
    cat ${scoringdir}/wer_details/per_utt \| \
    utils/scoring/wer_ops_details.pl --special-symbol "'***'" \| \
    sort -b -i -k 1,1 -k 4,4rn -k 2,2 -k 3,3 \> ${scoringdir}/wer_details/ops || exit 1;
  
  $cmd ${scoringdir}/log/wer_bootci.log \
    compute-wer-bootci --mode=present \
      ark:${scoringdir}/test.filt.txt ark:${scoringdir}/penalty_$best_wip/$best_lmwt.combine.filt.txt \
      '>' ${scoringdir}/wer_details/wer_bootci || exit 1;
  
  oov_rate=$(python local/get_oov_rate.py ${scoringdir}/test.filt.txt ${symtab})
  echo "OOV Rate: ${oov_rate}"
  echo ${oov_rate} > ${scoringdir}/wer_details/oov_rate 
  find $dir -name "wer_filt_[0-9]*" | xargs -I {} grep WER {} | ./utils/best_wer.sh
  grep WER $dir/wer_dup_filt*| ./utils/best_wer.sh
  grep WER $dir/wer_nodup_filt* | ./utils/best_wer.sh
fi


