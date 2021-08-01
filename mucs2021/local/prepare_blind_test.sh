#!/bin/bash

lang=hindi
data=/export/corpora5/MultilingualCodeSwitching2021/codeswitching/blindtest #/path/to/blindtest

mkdir -p data
cp -r ${data}/${lang}/files data/blindtest

for i in blindtest; do
  mv data/${i}/wav.scp data/${i}/wav.scp.bk
  awk -v var=${data}/${lang} '{print $1, var"/"$2}' data/${i}/wav.scp.bk > data/${i}/wav.scp
  ./utils/fix_data_dir.sh data/${i}
done
exit 0;
