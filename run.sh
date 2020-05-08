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
  --parameters="save_summary=false,"`
               `"feature_size=300,predicate_size=100,hidden_size=400,filter_size=800,"`
               `"residual_dropout=0.5,num_hidden_layers=8,attention_dropout=0.4,"`
               `"relu_dropout=0.4,batch_size=128,optimizer=adadelta,initializer=orthogonal,"`
               `"initializer_gain=1.0,train_steps=60000,"`
               `"learning_rate_schedule=piecewise_constant_decay,"`
               `"learning_rate=1,warmup_steps=4000,plot_step=1000,"`
               `"learning_rate_values=[1.0,0.5,0.25],learning_rate_boundaries=[40000,50000],"`
               `"save_checkpoint_steps=1000,early_stopping=10,save_checkpoint_steps=1500,"`
               `"device_list=$DEVICE,clip_grad_norm=1.0,script=validation.sh,"`
               `"buffer_size=512,"
# ,embedding=$EMBPATH/glove.6B.100d.txt
#[1.0,0.5,0.25,],[400000,50000],
