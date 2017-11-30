import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch import nn


def one_hot(seq_batch, depth):
    """Adds a new one-hot dim using the indices in the last dim
       seq_batch.size() should be [seq, batch] or [batch,]
       return size() would be [seq, batch, depth] or [batch, depth]
    """
    out = torch.zeros(seq_batch.size() + torch.Size([depth]))
    dim = len(seq_batch.size())
    index = seq_batch.view(seq_batch.size() + torch.Size([1]))
    return out.scatter_(dim, index, 1)

#
# def sample_gumbel(shape, eps=1e-20):
#     """Sample from Gumbel(0, 1)"""
#     u = torch.rand(shape)
#     g = -torch.log(-torch.log(u + eps) + eps)
#     return g
#     #return Variable(g, requires_grad=False)
#
#
# def gumbel_softmax_sample(logits, temperature=1):
#     """ Draw a sample from the Gumbel-Softmax distribution"""
#     y = logits + sample_gumbel(logits.shape)
#     y = F.log_softmax(y.view(-1, logits.size(-1)) / temperature)
#     return y.view_as(logits)
#
#
# def gumbel_softmax(logits, temperature=1, hard=True):
#     """Sample from the Gumbel-Softmax distribution and optionally discretize.
#     Args:
#         logits: [batch_size, n_class] un-normalized log-probs
#         temperature: non-negative scalar
#         hard: if True, take argmax, but differentiate w.r.t. soft sample y
#     Returns:
#         [batch_size, n_class] sample from the Gumbel-Softmax distribution.
#         If hard=True, then the returned sample will be one-hot, otherwise it will
#         be a probabilitiy distribution that sums to 1 across classes
#     """
#     y = gumbel_softmax_sample(logits, temperature)
#     if hard:
#         y_max_val, y_hard = y.max(1)
#         y_hard = one_hot(y_hard.data, logits.data.shape[1])
#         y = Variable(y_hard - y.data, requires_grad=False) + y # XXX
#     return y


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
    dims = len(logits.size())
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


class FactorizedEmbeddings(nn.Module):

    def __init__(self, emb_size, n_dic, n_codes):
        super().__init__()
        self.emb_size = emb_size
        self.n_codes = n_codes
        self.n_dic = n_dic
        self.tables = torch.nn.Parameter(torch.nn.init.xavier_uniform(
                torch.Tensor(n_dic, n_codes, emb_size)))

    def forward(self, x):
        """x has shape `batch x n_dict x n_codes"""
        table = self.tables
        prod = x @ table
        code = torch.sum(prod, 0)
        return code


class EmbeddingCompressor(nn.Module):
    def __init__(self, emb_size, n_tables, n_codes, temp, hard):
        """
           n_tables:   (M)
           n_codes: (K)
        """
        super().__init__()
        self.emb_size = emb_size
        self.n_codes = n_codes
        self.n_tables = n_tables
        self.temp = temp
        self.hard = hard
        mk_half = math.floor(n_tables * n_codes / 2)
        self.linear1 = torch.nn.Linear(emb_size, mk_half)
        self.linear2 = torch.nn.Linear(mk_half, n_tables * n_codes)
        self.embeddings = FactorizedEmbeddings(emb_size, n_tables, n_codes)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # x = batch x emb_size
        h = nn.Tanh()(self.linear1(x))

        # h = batch x (n_tables * n_codes / 2)
        a = nn.functional.softplus(self.linear2(h))
        #a = a.view(self.n_tables, -1, self.n_codes)

        # a = batch x (n_tables * n_codes)
        #print('a', a.data.shape)

        d = gumbel_softmax(a, tau=self.temp, hard=self.hard)
        d = d.view(self.n_tables, -1, self.n_codes)
        #d = d.view(self.n_tables, -1, self.n_codes)
        # print('d', d.size())
        # print(d.data)
        #print(torch.sum(d.data, -1))

        return self.embeddings(d)

