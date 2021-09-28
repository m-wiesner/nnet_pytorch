#!/bin/bash

. ./path.sh

stage=0

. ./utils/parse_options.sh
if [ $# -lt 5 ]; then
  echo "Usage: ./local/prepare_multilingual_data.sh <olang> <odata> <workdir> <dict1> <data1> [<dict2> <data2> ...]"
  exit 1;
fi

olang=$1
odata=$2
workdir=$3

while (( "$#" )); do
  dictionaries+=("$4")
  datadirs+=("$5")
  shift;
  shift;
done

mkdir -p $workdir

###############################################################################
# Combine all of the language specific lexicons to form one global dict and
# lang directory
###############################################################################

if [ $stage -le 0 ]; then
  echo ----------------------------------
  echo "Combining dict across languages on " `date`
  echo ----------------------------------
  
  dict=${workdir}/lexicon
  mkdir -p ${dict}
  # This script combines the lexicons to utlimately create a new lang dir.
  # Since we don't want the same words across languages to share pronunciations
  # we simply map each word to an integer.
  LC_ALL= python ./local/combine_lexicons.py ${dict} ${dictionaries[@]}
  echo -e "<silence>\tSIL\n<unk>\t<oov>\n<noise>\t<sss>\n<v-noise>\t<vns>" \
    > ${dict}/silence_lexicon.txt
  ./local/prepare_dict.py --silence-lexicon ${dict}/silence_lexicon.txt ${dict}/lexicon.txt ${dict}
  ./utils/prepare_lang.sh --share-silence-phones true ${dict} "<unk>" ${dict}/tmp.lang ${olang} 

  ./utils/combine_data.sh ${odata} ${datadirs[@]}
  
  # Remap words in training directories according to the wordmap.* files
  # created in ./local/combine_lexicons.py
  i=0
  for l in ${dictionaries[@]}; do
    datadir=${datadirs[$i]}
    wordmap=${dict}/`grep ${l} ${dict}/word_maps.scp | awk '{print $2}'`
    #echo "Converting datadir, ${datadir}, according to ${l} using ${wordmap}"
    cat ${datadir}/text | ./utils/sym2int.pl -f 2- ${wordmap}
    i=$(($i + 1))
  done > ${odata}/text 

  ./utils/fix_data_dir.sh ${odata}
  cut -d' ' -f2 ${odata}/segments | cut -d'_' -f3 |\
    paste -d'_' - ${odata}/segments > ${odata}/segments_

  cut -d' ' -f2 ${odata}/segments | cut -d'_' -f3 |\
    paste -d'_' - ${odata}/text > ${odata}/text_

  awk -F'[ _]' '{print $1"_"$2"_"$3"_"$4"_"$5"_"$6" "$1"_"$2"_"$3}' ${odata}/segments_ > ${odata}/utt2spk_

  rm ${odata}/spk2utt
  for f in segments text utt2spk; do
    mv ${odata}/${f} ${odata}/.${f}.bk
    mv ${odata}/${f}_ ${odata}/${f}
  done
  ./utils/utt2spk_to_spk2utt.pl ${odata}/utt2spk > ${odata}/spk2utt
  ./utils/fix_data_dir.sh ${odata}
  awk -F'[ _]' '{print $1"_"$2"_"$3"_"$4"_"$5"_"$6" "$1}' ${odata}/utt2spk > ${odata}/utt2lang 
fi

    
