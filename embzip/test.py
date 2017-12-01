import torch
from torch.autograd import Variable

from parameterized import parameterized

from embzip.data import load_embeddings_txt, make_vocab
from embzip.model import gumbel_softmax, EmbeddingCompressor


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