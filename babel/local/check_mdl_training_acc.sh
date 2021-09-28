#!/bin/bash

intval=20
offset=10
smooth=0.99
field=CrossEntropyAcc

. ./utils/parse_options.sh 

mdl=$1
sepoch=$2
eepoch=$3

for l in `seq ${sepoch} ${eepoch}`; do
  grep -o "${field}: *[-.0-9.]*" ${mdl}/log/train.${l}.1.log
done | \
  awk -v lambda=${smooth} '(NR==1){avg=$2; print avg; next} (NR>1){avg=lambda*avg + (1-lambda)*$2; print avg}' |\
  awk -v inval=${intval} -v offset=${offset} '((NR-offset)%inval == 0)'

