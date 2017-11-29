import torch
from torch.autograd import Variable

from parameterized import parameterized

from embzip.data import load_embeddings_txt, make_vocab
from embzip.model import FactorizedEmbeddings,one_hot, gumbel_softmax, EmbeddingCompressor

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
           vocab['embeddings'].data.shape[0]


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
    #(7, 2, 3, [1, 1]),
    #(7, 2, 3, [[1, 1], [2, 2]]),
    (7, 4, 3, [[1, 1, 1, 1], [2, 2, 2, 2]]),
])
def test_factorized_embeddings(emb_size, n_dic, n_codes, indices):
    embs = FactorizedEmbeddings(emb_size, n_dic, n_codes)

    # set each table row to its index value
    for _, table in enumerate(embs.tables):
        for j in range(table.size(0)):
            table.data[j] = j

    # build input one-hot matrix
    indices = torch.LongTensor(indices)
    if len(indices.size()) == 1:
        indices = indices[None, :]

    sums = torch.sum(indices, 1)[:, None].repeat(1, emb_size)
    print(sums)

    o = one_hot(indices.transpose(0, 1).contiguous(), n_codes)
    print(o.size())
    out = embs(Variable(o))

    print(out.size())
    print(out)
    assert torch.equal(out.data.long(), sums)


@parameterized([
    (None, 5, 4),
    (2, 5, 4)
])
def test_gumbel_softmax(batch, n_dic, n_codes):
    size = torch.Size([n_dic, n_codes])
    if batch is not None:
        size = torch.Size([batch]) + size

    out_size = size[:-1]

    logits = Variable(1 * torch.Tensor(size).uniform_())

    gsm = gumbel_softmax(logits, temperature=1, hard=False)

    assert torch.equal(torch.ones(out_size), torch.sum(gsm, -1).data)


@parameterized([
    (2, 5, 2, 4)
])
def test_compressor(batch, emb_size, n_dic, n_codes):
    ec = EmbeddingCompressor(emb_size, n_dic, n_codes)
    inp = Variable(torch.Tensor(batch, emb_size).uniform_())
    out = ec(inp)
    print(out)