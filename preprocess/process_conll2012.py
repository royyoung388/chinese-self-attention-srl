import cgitb
import os
import random
import sys

from subword import *

cgitb.enable(format="text")

COUNT = 0

input_data_path = sys.argv[1]
output_dir = sys.argv[2]
output_file = sys.argv[3]
output_props_file = sys.argv[4]
output_propid_file = sys.argv[5]
output_domains_file = sys.argv[6]
max_count = int(sys.argv[7]) if len(sys.argv) > 7 and int(sys.argv[7]) > 0 else 100000
if len(sys.argv) > 8:
    exclude_labels = sys.argv[8].split(";")
    exclude_labels = ["B-" + label for label in exclude_labels] + ["I-" + label for label in exclude_labels]
    exclude_labels = set(exclude_labels)
else:
    exclude_labels = None

output = os.path.join(output_dir, output_file)
output_props = os.path.join(output_dir, output_props_file)
output_propid = os.path.join(output_dir, output_propid_file)
output_domains = os.path.join(output_dir, output_domains_file)

fout = open(output, 'w', encoding='utf-8')
fout_props = open(output_props, 'w', encoding='utf-8')
fout_propid = open(output_propid, 'w', encoding='utf-8')
fout_domains = open(output_domains, 'w', encoding='utf-8')

if exclude_labels:
    exclude_output_dir = os.path.join(output_dir, 'exclude')
    if not os.path.exists(exclude_output_dir):
        os.mkdir(exclude_output_dir)

    exclude_output = os.path.join(exclude_output_dir, output_file)
    exclude_output_props = os.path.join(exclude_output_dir, output_props_file)
    exclude_output_propid = os.path.join(exclude_output_dir, output_propid_file)
    exclude_output_domains = os.path.join(exclude_output_dir, output_domains_file)

    exclude_fout = open(exclude_output, 'w', encoding='utf-8')
    exclude_fout_props = open(exclude_output_props, 'w', encoding='utf-8')
    exclude_fout_propid = open(exclude_output_propid, 'w', encoding='utf-8')
    exclude_fout_domains = open(exclude_output_domains, 'w', encoding='utf-8')

total_props = 0
total_props2 = 0
total_sents = 0
total_sents2 = 0

prev_words = ''
domain = ''
dpath = []

doc_counts = 0
v_counts = 0
ner_counts = 0

words = []
props = []
props_line = []
tags = []
spans = []
all_props = []


def print_new_sentence(fout, fout_props, fout_propid, fout_domains):
    global total_props
    global total_props2
    global total_sents
    global words
    global props
    global props_line
    global tags
    global spans
    global all_props
    global domain

    ''' ALso output sentences without any predicates '''
    # if len(props) > 0:
    total_props += len(props)
    total_sents += 1
    assert len(props) == len(tags)

    propid_labels = ['O' for _ in words]
    for t in range(len(props)):
        assert len(tags[t]) == len(words)
        assert tags[t][props[t]] == "B-V"
        fout.write(str(props[t]) + " " + " ".join(words) + " ||| " + " ".join(tags[t]) + "\n")
        propid_labels[props[t]] = 'V'
        fout_domains.write(domain + '\n')

    fout_props.write(''.join(props_line))
    fout_props.write('\n')
    fout_propid.write(" ".join(words) + " ||| " + " ".join(propid_labels) + "\n")
    total_props2 += len(all_props)


for root, dirs, files in os.walk(input_data_path):
    for f in files:
        if not 'gold_conll' in f and not 'gold_parse_conll' in f:
            continue

        # randomly choose files
        if random.random() > max_count / 1350:
            continue

        if COUNT >= max_count:
            break
        else:
            COUNT += 1

        # print(root, dirs, f)
        dpath = root.split('/')
        domain = '_'.join(dpath[dpath.index('annotations') + 1:-1])
        fin = open(root + "/" + f, mode='r', encoding='utf8')
        # flist_out.write(f + '\n')
        doc_counts += 1
        for line in fin:
            line = line.strip().upper()
            if line == '':
                joined_words = " ".join(words)
                # if joined_words == prev_words:
                #  print "Skipping dup sentence in: ", root, f
                # else:
                prev_words = joined_words
                total_sents2 += 1

                print_new_sentence(fout, fout_props, fout_propid, fout_domains)
                if exclude_labels and len(set([e for t in tags for e in t]) & exclude_labels) > 0:
                    print_new_sentence(exclude_fout, exclude_fout_props, exclude_fout_propid, exclude_fout_domains)

                words = []
                props = []
                props_line = []
                tags = []
                spans = []
                all_props = []
                continue

            if line[0] == "#":
                prev_words = ""
                if len(words) > 0:
                    total_sents2 += 1

                    print_new_sentence(fout, fout_props, fout_propid, fout_domains)
                    if exclude_labels and len(set([e for t in tags for e in t]) & exclude_labels) > 0:
                        print_new_sentence(exclude_fout, exclude_fout_props, exclude_fout_propid, exclude_fout_domains)

                    words = []
                    props = []
                    props_line = []
                    tags = []
                    spans = []
                    all_props = []
                continue

            info = line.split()
            word = subword(info[3])
            # try:
            #     word = info[3]
            # except UnicodeEncodeError:
            #     print(root, dirs, f)
            #     print(info[3])
            #     word = "*UNKNOWN*"

            words.append(word)
            idx = len(words) - 1
            if idx == 0:
                tags = [[] for _ in info[11:-1]]
                spans = ["" for _ in info[11:-1]]

            is_predicate = (info[7] != '-')
            is_verbal_predicate = False

            # in chinese DataSet, some file info[6] always is '-', have to change by label
            # lemma = info[6] if info[7] != '-' else '-'
            lemma = info[3] if '(V*)' in info[11:-1] else '-'
            info_re = []
            for i in info[11:-1]:
                i = i.replace('C-', '')
                i = i.replace('R-', '')
                i = i.replace('REL-SUP', 'REL')
                info_re.append(i)
            props_line.append(lemma + '\t' + '\t'.join(info_re) + '\n')
            # fout_props.write()

            for t in range(len(tags)):
                arg = info[11 + t]
                label = arg.strip("()*")

                # special
                if "(" in label:
                    print(os.path.join(root, f))
                    print(label)
                # label = label.split("(")[0]
                # C-* -> * , B-* -> *
                if label.startswith('C-') or label.startswith('R-'):
                    label = label[2:]
                # REL-SUP -> REL
                if label.startswith('REL-'):
                    label = label[:3]

                if "(" in arg:
                    tags[t].append("B-" + label)
                    spans[t] = label
                elif spans[t] != "":
                    tags[t].append("I-" + spans[t])
                else:
                    tags[t].append("O")
                if ")" in arg:
                    spans[t] = ""
                if "(V" in arg:
                    is_verbal_predicate = True
                    v_counts += 1

            if '(' in info[10]:
                ner_counts += 1

            if is_verbal_predicate:
                props.append(idx)
            if is_predicate:
                all_props.append(idx)

        fin.close()
        ''' Output last sentence.'''
        if len(words) > 0:
            total_sents2 += 1

            print_new_sentence(fout, fout_props, fout_propid, fout_domains)
            if exclude_labels and len(set([e for t in tags for e in t]) & exclude_labels) > 0:
                print_new_sentence(exclude_fout, exclude_fout_props, exclude_fout_propid, exclude_fout_domains)

            words = []
            props = []
            props_line = []
            tags = []
            spans = []
            all_props = []
    else:
        continue
    break

fout.close()
fout_props.close()
fout_propid.close()
fout_domains.close()
# flist_out.close()

print('documents', doc_counts)
print('all sentences', total_sents, total_sents2)
print('props', total_props)
print('verbal props:', v_counts)
print('ner counts:', ner_counts)
print('sentences', total_sents)
