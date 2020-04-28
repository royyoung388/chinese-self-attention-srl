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
  "${SRLPATH}/conll2012.train.txt" \
  "${SRLPATH}/conll2012.train.props.gold.txt" \
  "${SRLPATH}/conll2012.propid.train.txt" \
  "${SRLPATH}/conll2012.train.domains"

python preprocess/process_conll2012.py \
  "${DEV}" \
  "${SRLPATH}/conll2012.devel.txt" \
  "${SRLPATH}/conll2012.devel.props.gold.txt" \
  "${SRLPATH}/conll2012.propid.devel.txt" \
  "${SRLPATH}/conll2012.devel.domains"

python preprocess/process_conll2012.py \
  "${TEST}" \
  "${SRLPATH}/conll2012.test.txt" \
  "${SRLPATH}/conll2012.test.props.gold.txt" \
  "${SRLPATH}/conll2012.propid.test.txt" \
  "${SRLPATH}/conll2012.test.domains"



