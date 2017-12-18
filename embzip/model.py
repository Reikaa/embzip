import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn


def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
    y = logits + Variable(gumbel_noise)
    #y = F.log_softmax(y.view(-1, logits.size(-1)) / tau)
    y = F.softmax(y.view(-1, logits.size(-1)) / tau)
    return y.view_as(logits)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class FactorizedEmbeddingsInput(nn.Module):
    def __init__(self, emb_table, vocab_map):
        '''
            emb_table: numpy array (V x E)
            vocab_map: {word -> [indices]}
        '''
        super().__init__()
        self.emb_tables = nn.Embedding(*emb_table.shape)
        self.emb_tables.weight.data.copy_(torch.from_numpy(emb_table))
        self.vocab_map = vocab_map
        self.index_map = torch.LongTensor(list(vocab_map.values()))

    def forward(self, indices):
        # B x S -> BS x M
        exp_indices = self.index_map[indices.data.view(-1)]
        # exp_indices = exp_indices.view(indices.size(0), indices.size(1), -1)

        # BS x M -> BS x M x E
        exp_embs = self.emb_tables(Variable(exp_indices))

        # BS x M x E -> BS x E
        exp_embs = exp_embs.sum(-2)

        # BS x E -> B x S x E
        exp_embs = exp_embs.view(indices.size(0), indices.size(1), -1)

        return exp_embs

    @classmethod
    def load_hdf5(cls, h5_file):
        print('Loading compressed embeddings in %s' % h5_file)
        f = h5py.File(h5_file, 'r')
        # embeddings
        embs = np.array(f['embeddings'])
        words = list(f['vocab'])
        indices = list(f['indices'])
        # print(embs.dtype, words, indices)
        vocab_map = {k: v.tolist() for k, v in zip(words, indices)}
        return cls(embs, vocab_map)



class FactorizedEmbeddingsOutput(nn.Module):
    def __init__(self, emb_table, vocab_map):
        '''
            emb_table: numpy array (V x E)
            vocab_map: {word -> [indices]}
        '''
        super().__init__()
        self.linear = nn.Linear(*emb_table.shape, bias=False)
        self.linear.weight.data.copy_(torch.from_numpy(emb_table))
        self.vocab_map = vocab_map
        self.index_map = torch.LongTensor(list(vocab_map.values()))

    def forward(self, x):
        # B x E -> B x MK
        mk_scores = self.linear(x)

        # B x MK -> B x V
        v_size = self.index_map.size(0)
        v_scores = [mk_scores[:, self.index_map[i]].sum(1)
                    for i in range(v_size)]
        return torch.stack(v_scores, 1)

    @classmethod
    def load_hdf5(cls, h5_file):
        print('Loading compressed embeddings in %s' % h5_file)
        f = h5py.File(h5_file, 'r')
        # embeddings
        embs = np.array(f['embeddings'])
        words = list(f['vocab'])
        indices = list(f['indices'])
        # print(embs.dtype, words, indices)
        vocab_map = {k: v.tolist() for k, v in zip(words, indices)}
        return cls(embs, vocab_map)


class EmbeddingCompressor(nn.Module):
    def __init__(self, emb_size, n_tables, n_codes, temp=1, hard=True):
        """
            emb_size: (E) embedding size (both in input and output)
            n_tables: (M) number of embedding tables
            n_codes:  (K) number of codes in each table
            temp:     temperature of gumbel-softmax
            hard:     switch between hard and soft gumbel-softmax
        """
        super().__init__()
        self.emb_size = emb_size
        self.n_codes = n_codes
        self.n_tables = n_tables
        self.temp = temp
        self.hard = hard
        mk_half = n_tables * n_codes // 2
        self.linear1 = nn.Linear(emb_size, mk_half)
        self.linear2 = nn.Linear(mk_half, n_tables * n_codes)
        self.emb_tables = nn.Linear(n_tables * n_codes, emb_size, bias=False)

    def get_indices(self, x):
        """Given a batch of embeddings of shape B x E
           returns a list-of-lists of non-zero indices, one per row
        """
        # B x E -> B x MK
        one_hot = self.encode(x)
        return [torch.nonzero(one_hot[i]).squeeze().cpu().tolist()
                for i in range(one_hot.size(0))]

    def encode(self, x):
        """Given a batch of embeddings of shape B x E
           returns a batch of one-hot embeddings of shape B x M x K
        """
        # B x E -> B x MK/2
        h = nn.Tanh()(self.linear1(x))

        # B x MK/2 -> B x MK
        a = nn.functional.softplus(self.linear2(h))

        # B x MK -> BM x K
        a = a.view(-1, self.n_codes)

        # BM x K -> BM x K
        d = gumbel_softmax(a, tau=self.temp, hard=self.hard)

        # BM x K -> B x MK
        return d.view(-1, self.n_tables * self.n_codes)

    def forward(self, x):
        # B x E -> B x MK
        one_hot = self.encode(x)

        # B x MK -> B x E
        return self.emb_tables(one_hot)
