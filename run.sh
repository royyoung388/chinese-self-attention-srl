#!/usr/bin/env bash
TAGGERPATH=/home/roy/Tagger
SRLPATH=$TAGGERPATH/data/srl/srlconll-1.1
DATAPATH=$TAGGERPATH/data/srl
EMBPATH=$TAGGERPATH/data/glove
DEVICE=[0]

export PYTHONPATH=$TAGGERPATH:$PYTHONPATH
export PERL5LIB="$SRLPATH/lib:$PERL5LIB"
export PATH="$SRLPATH/bin:$PATH"

python $TAGGERPATH/tagger/bin/trainer.py \
  --model deepatt \
  --subthread \
  --input $DATAPATH/conll2012.train.txt \
  --output train \
  --vocabulary $DATAPATH/vocab.txt $DATAPATH/label.txt \
  --parameters="save_summary=false,"`
               `"feature_size=300,predicate_size=100,hidden_size=400,filter_size=800,"`
               `"batch_size=2048,buffer_size=10240,num_hidden_layers=12,train_steps=500000,"`
               `"relu_dropout=0.2,residual_dropout=0.3,attention_dropout=0.2,lstm_dropout=0.2,"`
               `"optimizer=adadelta,initializer=orthogonal,"`
               `"learning_rate_schedule=piecewise_constant_decay,"`
               `"learning_rate=1,warmup_steps=4000,"`
               `"learning_rate_values=[1.0,0.1,0.01],learning_rate_boundaries=[300000,400000],"`
               `"save_checkpoint_steps=2000,early_stopping=10,plot_step=1000,"`
               `"device_list=$DEVICE,clip_grad_norm=1.0,script=validation.sh,embedding=$EMBPATH/embedding.txt"

#python $TAGGERPATH/tagger/bin/trainer.py \
#  --model deepatt \
#  --subthread \
#  --input $DATAPATH/exclude/conll2012.train.txt \
#  --output train \
#  --vocabulary $DATAPATH/vocab.txt $DATAPATH/label.txt \
#  --parameters="save_summary=false,"`
#               `"feature_size=3residual_dropout00,predicate_size=100,hidden_size=400,filter_size=800,"`
#               `"batch_size=2048,buffer_size=10240,num_hidden_layers=10,train_steps=120000,"`
#               `"relu_dropout=0.1,residual_dropout=0.2,attention_dropout=0.1,"`
#               `"optimizer=adadelta,initializer=orthogonal,initializer_gain=1.0,"`
#               `"learning_rate_schedule=piecewise_constant_decay,"`
#               `"learning_rate=1,warmup_steps=4000,"`
#               `"learning_rate_values=[1.0,0.5,0.25],learning_rate_boundaries=[100000,110000],"`
#               `"save_checkpoint_steps=2000,early_stopping=10,plot_step=1000,"`
#               `"device_list=$DEVICE,clip_grad_norm=1.0,embedding=$EMBPATH/embedding.txt,script=validation.sh,"
