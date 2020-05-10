import argparse
import cgitb
import os

import matplotlib.pyplot as plt
import seaborn
import torch

from tagger import utils, models, data
from tagger.bin import predictor
from tagger.bin.trainer import default_params, merge_params, import_params


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
    g.set_xticklabels(g.get_xticklabels(), rotation=90, horizontalalignment='right')
    # g.set_yticklabels(g.get_yticklabels(), rotation=45, horizontalalignment='right')


def get_model(args):
    model_cls = models.get_model(args.model)

    params = default_params()
    params = merge_params(params, model_cls.default_params())
    params = merge_params(params, predictor.default_params())
    params = import_params(args.dir, args.model, params)
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


def parse_args():
    msg = "Attention Visualization"
    parser = argparse.ArgumentParser(description=msg)

    msg = "model directory"
    parser.add_argument("dir", help=msg)
    msg = 'visual data'
    parser.add_argument('data', help=msg)
    msg = "model"
    parser.add_argument("--model", type=str, default='deepatt', help=msg)
    msg = "embedding"
    parser.add_argument("--embedding", type=str, default="", help=msg)
    return parser.parse_args()


def main(args):
    checkpoint = utils.best_checkpoint(os.path.join(args.dir, 'best'))
    model, params = get_model(args)
    model.eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])

    # Decoding
    dataset = data.get_dataset(args.data, "infer", params)

    if args.embedding:
        embedding = data.load_glove_embedding(args.embedding)
    else:
        embedding = None

    for i, features in enumerate(dataset):
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
            plt.savefig('attention_' + str(i) + '.png')


if __name__ == '__main__':
    cgitb.enable(format='text')
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['simhei']
    seaborn.set(font='SimHei')
    main(parse_args())
