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
  --input $DATAPATH/conll2012.devel.txt \
  --checkpoint train \
  --model deepatt \
  --vocabulary $DATAPATH/vocab.txt $DATAPATH/label.txt \
  --parameters=device=$DEVICE \
  --output tmp.txt
#,embedding=$EMBPATH/glove.6B.100d.txt

python $TAGGERPATH/tagger/scripts/convert_to_conll.py tmp.txt $DATAPATH/conll2012.devel.props.gold.txt output
perl $SRLPATH/bin/srl-eval.pl $DATAPATH/conll2012.devel.props.* output
