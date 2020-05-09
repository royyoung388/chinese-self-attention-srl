#!/usr/bin/env bash
TAGGERPATH=/home/roy/Tagger
SRLPATH=$TAGGERPATH/data/srl/srlconll-1.1
DATAPATH=$TAGGERPATH/data/srl
EMBPATH=$TAGGERPATH/data/glove
DEVICE=0

export PYTHONPATH=$TAGGERPATH:$PYTHONPATH
export PERL5LIB="$SRLPATH/lib:$PERL5LIB"
export PATH="$SRLPATH/bin:$PATH"

python $TAGGERPATH/tagger/bin/predictor.py \
  --input $DATAPATH/conll2012.test.txt \
  --checkpoint train/best \
  --model deepatt \
  --vocabulary $DATAPATH/vocab.txt $DATAPATH/label.txt \
  --parameters=device=$DEVICE \
  --output decode.txt
#,embedding=$EMBPATH/glove.6B.100d.txt