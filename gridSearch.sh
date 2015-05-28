#!/bin/bash

if [[ ! $# -eq 6 ]]; then
  echo "Usage: gridSearch.sh <train_signal_file> <train_rel_file> <test_signal_file> <task> [out_file] <test_rel_file>"
  exit
fi

train_signal_file=$1
train_rel_file=$2
test_signal_file=$3
task=$4
out_file=$5
test_rel_file=$6

#ant

#echo ""
#echo "# Executing: java -cp bin:lib/weka.jar cs276.pa4.Learning2Rank $train_signal_file $train_rel_file $test_signal_file $task $out_file $test_rel_file"
java -cp bin:lib/weka.jar cs276.pa4.Learning2Rank $train_signal_file $train_rel_file $test_signal_file $task $out_file $test_rel_file

