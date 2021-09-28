#!/bin/bash

delim=" "

if [ $# -ne 2 ]; then
  echo >&2 "Usage: ./local/phoneset_diff.sh <lex1> <lex2> > missing"
  echo >&2 "  Assumes, both lexicons are produced by local/prepare_lexicon.pl."
  echo >&2 "  -------------------"
  echo >&2 "  Find the phones in 1 that are not in 2."
  exit 1;
fi


lex1=$1
lex2=$2

export LC_ALL=C
comm -23 <(cut -d"$delim" -f2- $lex1 | tr " " "\n" | sed 's/^ *$//g;s/\t//g' | sort -u)\
         <(cut -d"$delim" -f2- $lex2 | tr " " "\n" | sed 's/^ *$//g;s/\t//g' | sort -u)  
unset LC_ALL
