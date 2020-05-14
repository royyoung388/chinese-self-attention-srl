import math
import os
import random

from processor import Processor

TRAIN = "/home/roy/conll-2012/v4/data/train/data/chinese/annotations"
DEV = "/home/roy/conll-2012/v4/data/development/data/chinese/annotations"
TEST = "/home/roy/conll-2012/v9/data/test/data/chinese/annotations"
proportion = '6:1:1'
output_dir = "/home/roy/Tagger/data/srl"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

all_files = []


def walk(path):
    if path is None:
        return
    for root, dirs, files in os.walk(path):
        for f in files:
            if not 'gold_conll' in f and not 'gold_parse_conll' in f:
                continue
            else:
                all_files.append(os.path.join(root, f))


walk(TRAIN)
walk(DEV)
walk(TEST)

random.shuffle(all_files)

p = proportion.split(':')
p = list(map(int, p))
all = len(all_files)

train_files = all_files[: math.ceil(p[0] / sum(p) * all)]
val_files = all_files[math.ceil(p[0] / sum(p) * all): math.ceil((p[0] + p[1]) / sum(p) * all)]
test_files = all_files[math.ceil((p[0] + p[1]) / sum(p) * all):]

train_processor = Processor(output_dir, "conll2012.train.txt",
                            "conll2012.train.props.gold.txt",
                            "conll2012.train.propid.txt",
                            "conll2012.train.domains",
                            "ARG3;ARG4;ARGM-BNF;ARGM-DGR;ARGM-EXT;ARGM-FRQ;ARGM-NEG;ARGM-PRD;ARGM-TPC;REL")
train_processor.process(train_files)

val_processor = Processor(output_dir, "conll2012.devel.txt",
                          "conll2012.devel.props.gold.txt",
                          "conll2012.devel.propid.txt",
                          "conll2012.devel.domains")
val_processor.process(val_files)

test_processor = Processor(output_dir, "conll2012.test.txt",
                           "conll2012.test.props.gold.txt",
                           "conll2012.test.propid.txt",
                           "conll2012.test.domains")
test_processor.process(test_files)
