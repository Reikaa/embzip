import torch
from torch.autograd import Variable

from parameterized import parameterized

from embzip.data import load_embeddings_txt, make_vocab
from embzip.model import gumbel_softmax, EmbeddingCompressor
from embzip.data import load_embeddings_txt, make_vocab, load_hdf5, save_hdf5
from embzip.model import (gumbel_softmax, EmbeddingCompressor,
                          FactorizedEmbeddingsOutput,
                          FactorizedEmbeddingsInput)


@parameterized([
    ('data/glove.6B.300d.txt', 100),
])
def test_glove_txt(path, maxsize):
    gloves = load_embeddings_txt(path, maxsize)
    assert len(gloves) <= maxsize
    for k, v in gloves.items():
        assert isinstance(k, str)
        assert isinstance(v, torch.autograd.Variable)
        assert v.data.shape[0] == 1


@parameterized([
    ('data/glove.6B.300d.txt', 100),
])
def test_vocab(path, maxsize):
    gloves = load_embeddings_txt(path, maxsize)
    vocab = make_vocab(gloves)

    assert len(vocab['idx2word']) == \
           len(vocab['word2idx']) == \
           vocab['emb_tables'].data.shape[0]


@parameterized([
    (None, 3, 4),
    (2, 3, 4),
])
def test_one_hot(batch, n_dic, n_codes):
    size = torch.Size([n_dic])
    if batch is not None:
        size = torch.Size([batch]) + size

    t = torch.LongTensor(size).random_() % n_codes

    o = one_hot(t, n_codes)
    print(o)

    exp_size = size + torch.Size([n_codes])
    assert exp_size == o.size()


@parameterized([
    (2, 3, 4)
])
def test_gumbel_softmax(b, m, k):

    logits = Variable(1 * torch.Tensor(b * m, k).uniform_())

    gsm = gumbel_softmax(logits, tau=1, hard=True)
    print(gsm)
    print(gsm.view(b, m * k))

    assert torch.equal(torch.ones(b*m), torch.sum(gsm, -1).data)


@parameterized([
    (2, 5, 2, 4)
])
def test_compressor(batch, emb_size, n_dic, n_codes):
    ec = EmbeddingCompressor(emb_size, n_dic, n_codes)
    inp = Variable(torch.Tensor(batch, emb_size).uniform_())
    out = ec(inp)
    print(out)


@parameterized([
    ('data/glove.6B.300d.txt', 100),
])
def test_save_hdf5(path, num_embs):
    gloves, _ = load_embeddings_txt(path, num_embs)
    ec = EmbeddingCompressor(300, 8, 16)

    orig_embs, orig_vocab_map = save_hdf5('foo.h5', ec, gloves)
    embs, vocab_map = load_hdf5('foo.h5')

    assert(np.array_equal(orig_embs, embs))
    assert orig_vocab_map == vocab_map


@parameterized([
    ([[0]], [[0, 1], [1, 2]]),
    ([[0, 1], [1, 2]], [[1, 2], [3, 2], [0, 3]]),
    ([[0, 1], [2, 1]], [[0, 1], [1, 2], [2, 3]])
])
def test_fe_input(indices, index_map):
    vocab_map = {i: index_map[i] for i in range(len(index_map))}
    emb_table = np.arange(10).reshape(5, 2)
    indices = Variable(torch.LongTensor(indices))
    print(indices)

    fei = FactorizedEmbeddingsInput(emb_table, vocab_map)
    out = fei(indices)
    print(out)

    # figure out expanded embeddings
    index_map_t = torch.LongTensor(index_map)
    index_map_t = torch.autograd.Variable(index_map_t)

    exp_emb_table = torch.nn.Embedding(*emb_table.shape)
    exp_emb_table.weight.data.copy_(torch.from_numpy(emb_table))

    final_emb_table = exp_emb_table(index_map_t).sum(-2).data
    et = torch.nn.Embedding(*final_emb_table.shape)
    et.weight.data.copy_(final_emb_table)

    exp_out = et(indices)
    print(exp_out)
    assert torch.equal(exp_out.data, out.data)

@parameterized([
    ([[1, 1]], [[0, 1], [1, 2]]),
    ([[0, 1], [1, 2]], [[1, 2], [3, 2], [0, 3]]),
    ([[0, 1]], [[0, 1], [1, 2], [2, 3]]),
    ([[0, 1], [1, 1]], [[0, 1], [1, 2], [2, 3]]),
    ([[0, 1], [1, 1], [1, 0]], [[0, 1], [1, 2], [2, 3]])
])
def test_fe_output(pe, index_map):
    e = 2
    mk = 5
    v = len(index_map)
    mk_table = np.arange(10).reshape(e, mk)
    mk_table_T = np.arange(10).reshape(mk, e)
    vocab_map = {i: index_map[i] for i in range(len(index_map))}
    feo = FactorizedEmbeddingsOutput(np.array(mk_table), vocab_map)

    print(mk_table_T)
    emb_table = np.vstack([mk_table_T[index_map[i]].sum(0) for i in range(v)])
    print('e', emb_table)

    pe = Variable(torch.Tensor(pe))

    out = feo(pe)

    out_exp = pe.data @ torch.from_numpy(emb_table).float().t()
    print('o', out_exp)

    assert torch.equal(out.data, out_exp)
