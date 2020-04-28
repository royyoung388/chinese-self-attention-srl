#!/usr/bin/env bash
SRLPATH=/home/roy/Tagger/data/srl/srlconll-1.1
TAGGERPATH=/home/roy/Tagger
DATAPATH=/home/roy/Tagger/data/srl
EMBPATH=/home/roy/Tagger/data/glove
DEVICE=0

export PYTHONPATH=$TAGGERPATH:$PYTHONPATH
export PERL5LIB="$SRLPATH/lib:$PERL5LIB"
export PATH="$SRLPATH/bin:$PATH"

python $TAGGERPATH/tagger/bin/predictor.py \
  --input $DATAPATH/conll2012.devel.txt \
  --checkpoint train \
  --model deepatt \
  --vocab $DATAPATH/vocab.txt $DATAPATH/label.txt \
  --parameters=device=$DEVICE \
  --output tmp.txt
#,embedding=$EMBPATH/glove.6B.100d.txt

python $TAGGERPATH/tagger/scripts/convert_to_conll.py tmp.txt $DATAPATH/conll2012.devel.props.gold.txt output
perl $SRLPATH/bin/srl-eval.pl $DATAPATH/conll2012.devel.props.* output
