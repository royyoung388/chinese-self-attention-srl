# 基于Self-Attention的汉语语义角色标注  
本文模型基于[Deep Semantic Role Labeling with Self-Attention](https://github.com/XMUNLP/Tagger)

# 数据预处理
## 获取数据
在LDC上获取ontonotes 5.0数据 https://catalog.ldc.upenn.edu/LDC2013T19  

## 将数据转化为Conll格式
依照这篇教程将数据转为Conll格式 http://conll.cemantix.org/2012/data.html

## 数据处理脚本
修改 make_conll2012_data.sh 脚本的变量.
```shell script
# 训练集,开发集,测试集的路径
TRAIN=".../conll-2012/v4/data/train/data/chinese/annotations"
DEV=".../conll-2012/v4/data/development/data/chinese/annotations"
TEST=".../conll-2012/v9/data/test/data/chinese/annotations"
```

然后运行该脚本
```shell script
make_conll2012_data.sh
```

运行后,会在 data/srl 目录下生成.txt数据文件,以及exclude文件夹(单独包含了脚本中指定的特殊标签)  
处理后的数据格式如下
```text
2 My cats love hats . ||| B-A0 I-A0 B-V B-A1 O
```
## 生成字典
```shell script
# limit 代表字典的大小, lower 代表小写
python tagger/scripts/build_vocab.py --limit 20000 --lower data/srl/conll2012.train.txt data/srl
```

# 运行
## 修改脚本
修改 run.sh validation.sh 脚本变量参数
```shell script
TAGGERPATH=本项目根目录
```

并根据需要修改`parameters`参数

##运行
```shell script
./run.sh
```

##验证
```shell script
./validation.sh
```

# 结果
## Attention可视化
将需要可视化的数据复制到visual.txt中,然后运行
```shell script
python tagger/scripts/visualization.py train visual.txt --embedding EMBEDDING
```

## 使用预训练向量
注意文件开头如果是字典长度的信息,则该行需要删除