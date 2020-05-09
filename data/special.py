import collections
import os


def read(path):
    with open(path, 'r', encoding='utf-8') as f_in, open('special.txt', 'a', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            for c in line:
                # not chinese
                if c < u'\u4e00' or c > u'\u9fa5':
                    f_out.write(line)
                    f_out.write('\n')
                    break


def find(string, path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'gold_conll' not in file:
                continue
            p = os.path.join(root, file)
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip()
                    if string in parts:
                        print(p)
                        break


def compare_label(path1, path2):
    counter1 = collections.Counter()
    counter2 = collections.Counter()

    with open(path1, 'r', encoding='utf-8') as f1:
        for line in f1:
            counter1.update([line.strip()])
    with open(path2, 'r', encoding='utf-8') as f2:
        for line in f2:
            counter2.update([line.strip()])

    print(counter1 - counter2)
    print(counter2 - counter1)


def compare_length(path1, path2):
    with open(path1, 'r', encoding='utf-8') as f1, open(path2, 'r', encoding='utf-8') as f2:
        for line1, line2 in zip(f1, f2):
            len1 = len(line1.strip().split())
            len2 = len(line2.strip().split())
            print(len1 - len2)


def exclude_percent(path, exclude):
    count = 0
    all = 0
    exclude = exclude.split(';')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            labels = line.split("|||")[-1].split()
            for l in labels:
                if l == 'O':
                    continue
                all += 1
                if l[2:] in exclude:
                    count += 1
    print(count, '/', all)
    print('%.2f' % (count / all * 100.))

if __name__ == '__main__':
    # read('train/word_vocab.txt')
    # read('dev/word_vocab.txt')
    # read('test/word_vocab.txt')
    # find('阿吉兹', 'E:/SRL/conll-2012/v4/data/train/data/chinese')
    # find('rel', 'E:/SRL/conll-2012/v4/data/development/data/chinese')
    # find('rel', 'E:/SRL/conll-2012/v9/data/test/data/chinese')
    # compare_label('train/label_vocab.txt', 'test/label_vocab.txt')
    # compare_label('train/label_vocab.txt', 'dev/label_vocab.txt')
    # compare_length('dev/label.txt', 'dev/label_outphjm*.txt')
    exclude_percent('srl/exclude/conll2012.train.txt',
                    "ARG3;ARG4;ARGM-BNF;ARGM-DGR;ARGM-EXT;ARGM-FRQ;ARGM-NEG;ARGM-PRD;ARGM-TPC;REL")
