import math
import torch
import numpy as np
import h5py

from collections import OrderedDict


def load_embeddings_txt(path, max=None, vsize=0):
    train = OrderedDict()
    valid = OrderedDict()
    with open(path, 'rt', encoding='utf-8') as ef:
        for i, line in enumerate(ef):
            if i >= max+vsize:
                break
            tokens = line.split()
            word = tokens[0]
            # 1 x EmbSize
            vector = np.array(tokens[1:], dtype=np.float32)[None, :]
            if i < max:
                train[word] = vector
            if i > max:
                valid[word] = vector
    return train, valid


def make_vocab(word2vector):
    if not word2vector:
        return None
    idx2word = []
    word2idx = {}
    all_vars = []
    for word, vector in word2vector.items():
        word2idx[word] = len(idx2word)
        idx2word.append(word)
        all_vars.append(torch.from_numpy(vector))
    embeddings = torch.cat(all_vars, 0)
    embeddings = torch.autograd.Variable(embeddings, requires_grad=False)
    return dict(idx2word=idx2word,
                word2idx=word2idx,
                embeddings=embeddings,
                n_embs=embeddings.size(0),
                emb_size=embeddings.size(1))


def print_compression_stats(vocab, n_tables, n_codes):
    n_words = vocab['n_embs']
    emb_size = vocab['emb_size']
    orig_mb = n_words * emb_size * 4 / 2 ** 20
    comp_mb = n_tables * n_codes * emb_size * 4 / 2 ** 20
    ratio = -   100 * (orig_mb - comp_mb) / orig_mb

    print('[ORIG] {:4} words with embedding size {:3}: {:3.3} MB'.format(n_words, emb_size, orig_mb))
    print('[COMP] {:4} words ({:2} tables, {:3} codes):  {:3.3} MB ({:.2f}%)'.format(n_words, n_tables, n_codes, comp_mb, ratio))


def check_training(old_params, new_params):
    if old_params is None:
        return
    old_sums = [p.sum().data[0] for p in old_params]
    new_sums = [p.sum().data[0] for p in new_params]
    for old, new in zip(old_sums, new_sums):
        if old == new:
            print(old_sums)
            print(new_sums)
            print('WARNING: some parameter did not seem to change!')


def euclidian_dist(x, y):
    return math.sqrt((x - y).pow(2).sum())


def dump_reconstructed_embeddings(out_file, model, emb_dict):
    print('Saving reconstructed emb_tables for train set in %s' % out_file)
    with open(out_file, 'wt') as f:
        for word, emb in emb_dict.items():
            print(word)
            emb_comp = model(emb).squeeze().cpu().data.numpy().tolist()
            for x in emb_comp:
                word += ' ' + str(x)
            f.write(word + '\n')


def save_hdf5(out_file, model, emb_dict):
    print('Saving compressed embedding in %s' % out_file)
    # embeddings
    f = h5py.File(out_file, "w")
    f.create_dataset("embeddings", data=model.emb_tables.W.data)
    # words
    words = list(emb_dict.keys)

    # indices

