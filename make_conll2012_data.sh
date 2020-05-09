#! /bin/bash
TRAIN="/home/roy/conll-2012/v4/data/train/data/chinese/annotations"
DEV="/home/roy/conll-2012/v4/data/development/data/chinese/annotations"
TEST="/home/roy/conll-2012/v9/data/test/data/chinese/annotations"

#TRAIN="/home/roy/conll-formatted-ontonotes-5.0/data/train/data/english/annotations"
#DEV="/home/roy/conll-formatted-ontonotes-5.0/data/development/data/english/annotations"
#TEST="/home/roy/conll-formatted-ontonotes-5.0/data/conll-2012-test/data/english/annotations"

SRLPATH="./data/srl"

if [ ! -d $SRLPATH ]; then
  mkdir -p $SRLPATH
fi

python preprocess/process_conll2012.py \
  "${TRAIN}" \
  "${SRLPATH}" \
  "conll2012.train.txt" \
  "conll2012.train.props.gold.txt" \
  "conll2012.propid.train.txt" \
  "conll2012.train.domains" \
  -1 \
  "ARG3;ARG4;ARGM-BNF;ARGM-DGR;ARGM-EXT;ARGM-FRQ;ARGM-NEG;ARGM-PRD;ARGM-TPC;REL"

python preprocess/process_conll2012.py \
  "${DEV}" \
  "${SRLPATH}" \
  "conll2012.devel.txt" \
  "conll2012.devel.props.gold.txt" \
  "conll2012.propid.devel.txt" \
  "conll2012.devel.domains" \

python preprocess/process_conll2012.py \
  "${TEST}" \
  "${SRLPATH}" \
  "conll2012.test.txt" \
  "conll2012.test.props.gold.txt" \
  "conll2012.propid.test.txt" \
  "conll2012.test.domains" \



