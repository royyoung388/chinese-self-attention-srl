import cgitb
import os

import matplotlib.pyplot as plt
import seaborn
import torch

from tagger import utils, models, data
from tagger.bin import predictor
from tagger.bin.trainer import default_params, merge_params, import_params

cgitb.enable(format='text')
PATH = 'train'
DATA = 'visual.txt'
EMBEDDING = ''

plt.rcParams['font.sans-serif'] = ['simhei']


def normal(data: torch.Tensor):
    data[range(len(data)), range(len(data))] = 0
    m = torch.min(data)
    k = 1.0 / (torch.max(data) - m)
    data = k * (data - m)
    data[range(len(data)), range(len(data))] = 1
    return data


def draw(data, x, y, ax):
    g = seaborn.heatmap(data, cmap="Greys", xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
                        cbar=False, ax=ax)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    # g.set_yticklabels(g.get_yticklabels(), rotation=45, horizontalalignment='right')


def get_model():
    model_cls = models.get_model('deepatt')

    params = default_params()
    params = merge_params(params, model_cls.default_params())
    params = merge_params(params, predictor.default_params())
    params = import_params(PATH, 'deepatt', params)
    params.decode_batch_size = 1
    src_vocab, src_w2idx, src_idx2w = data.load_vocabulary(params.vocab[0])
    tgt_vocab, tgt_w2idx, tgt_idx2w = data.load_vocabulary(params.vocab[1])

    params.vocabulary = {
        "source": src_vocab, "target": tgt_vocab
    }
    params.lookup = {
        "source": src_w2idx, "target": tgt_w2idx
    }
    params.mapping = {
        "source": src_idx2w, "target": tgt_idx2w
    }

    torch.cuda.set_device(0)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Create model
    model = model_cls(params).cuda()
    return model, params


checkpoint = utils.best_checkpoint(os.path.join(PATH, 'best'))
model, params = get_model()
model.eval()
model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])

# Decoding
dataset = data.get_dataset(DATA, "infer", params)

if EMBEDDING:
    embedding = data.load_glove_embedding(EMBEDDING)
else:
    embedding = None

for features in dataset:
    seqs = features[0]['inputs'].numpy().tolist()
    sent = [w.decode('utf-8') for s in seqs for w in s]
    # sent = ["\n".join(list(word)) for word in sent]

    features = data.lookup(features, "infer", params, embedding)

    labels = model.argmax_decode(features)

    for layer in range(7, 8):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        print("Encoder Layer", layer + 1)
        for h in range(8):
            draw(normal(model.encoder.layers[layer].self_attention.attention.weights[0, h].data.cpu()),
                 sent, sent if h == 0 else [], ax=axs[h // 4, h % 4])
        plt.savefig('attention.png')
        plt.show()
