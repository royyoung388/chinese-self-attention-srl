#!/usr/bin/env bash
SRLPATH=/home/roy/Tagger/data/srl/srlconll-1.1
TAGGERPATH=/home/roy/Tagger
DATAPATH=/home/roy/Tagger/data/srl
EMBPATH=/home/roy/Tagger/data/glove
DEVICE=[0]

export PYTHONPATH=$TAGGERPATH:$PYTHONPATH
export PERL5LIB="$SRLPATH/lib:$PERL5LIB"
export PATH="$SRLPATH/bin:$PATH"

python $TAGGERPATH/tagger/bin/trainer.py \
  --model deepatt \
  --input $DATAPATH/conll2012.train.txt \
  --output train \
  --vocabulary $DATAPATH/vocab.txt $DATAPATH/label.txt \
  --parameters="save_summary=false,feature_size=100,hidden_size=200,filter_size=800,"`
               `"residual_dropout=0.2,num_hidden_layers=10,attention_dropout=0.1,"`
               `"relu_dropout=0.1,batch_size=2048,optimizer=adadelta,initializer=orthogonal,"`
               `"initializer_gain=1.0,train_steps=100000,"`
               `"learning_rate_schedule=piecewise_constant_decay,"`
               `"learning_rate_values=[1.0,0.1,0.01,0.001,0.0001],"`
               `"learning_rate_boundaries=[2000,10000,50000,80000],device_list=$DEVICE,"`
               `"clip_grad_norm=1.0,script=validation.sh"
# ,embedding=$EMBPATH/glove.6B.100d.txt