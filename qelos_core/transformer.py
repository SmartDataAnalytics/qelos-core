import torch
from torch import nn
from torch.nn import Parameter
import math
import copy
import qelos_core as q


# region from huggingface github transformer
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, dim, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.e = e

    def forward(self, x, mask=None):
        u = q.masked_mean(x, -1, mask=mask, keepdim=True)
        s = q.masked_mean((x - u).pow(2), -1, mask=mask, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):       # adapted from Conv1D
    """
    1D convolution with window size 1 = multiple every vector in input with matrix
    """
    def __init__(self, indim, outdim):     # indim, outdim
        super(Conv1D, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.layer = torch.nn.Linear(indim, outdim, bias=True)
        nn.init.normal_(self.layer.weight, std=0.002)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x):       # (batsize, [seqlen, ...], indim)
        return self.layer(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, indim, bidir=False, numheads=None, attention_dropout=0., residual_dropout=0., scale=False):    # indim
        super(MultiHeadAttention, self).__init__()
        outdim = indim  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert outdim % numheads == 0    # ensure that indim supports the number of heads
        # self.register_buffer('mask', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.numheads = numheads
        self.scale = scale
        self.bidir = bidir
        self.qkv_proj = Conv1D(indim, outdim * 3)     # projects input to query, key and value
        self.vw_proj = Conv1D(indim, outdim)          # projects after-attention summaries
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(residual_dropout)

    def forward(self, x, mask=None):   # (batsize, seqlen, indim), (batsize, seqlen)
        new_x_shape = x.size()[:-1] + (3, self.numheads, x.size(-1) // self.numheads)
        qkv = self.qkv_proj(x).view(*new_x_shape)  # in Tensorflow implem: fct split_states
        # (batsize, seqlen, 3, numheads, dim_per_head)

        # ATTENTION
        # for every vector in the sequence for every head for every batch, multiply it with every head in the same sequence
        w = torch.einsum("bshd,bzhd->bhsz", (qkv[:, :, 0], qkv[:, :, 1]))
        # (batsize, numheads, seqlen, seqlen)
        v = qkv[:, :, 2]  # (batsize, seqlen, numheads, dim_per_head)
        if self.scale:
            w = w / math.sqrt(v.size(-1))  # scale attention weights by dimension of values
        wholemask = None
        if mask is not None:
            # w = w + torch.log(mask.float().view(mask.size(0), 1, mask.size(1), 1))
            wholemask = mask.float().view(mask.size(0), 1, 1, mask.size(1))
        if self.bidir is False:
            seqlen = w.size(-1)
            causality_mask = torch.tril(torch.ones(seqlen, seqlen, device=x.device))\
                .view(1, 1, seqlen, seqlen)
            wholemask = wholemask * causality_mask if wholemask is not None else causality_mask
            # * self.mask + -1e9 * (1 - self.mask)  # TF implem method: mask_attn_weights
        if wholemask is not None:
            w = w + torch.log(wholemask)
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        vw = torch.einsum("bhsz,bzhd->bshd", (w, v))  # (batsize, seqlen, numheads, dim_per_head)
        # end attention

        new_shape = vw.size()[:-2] + (vw.size(-2) * vw.size(-1),)
        vw = vw.contiguous().view(*new_shape)
        _vw = self.vw_proj(vw)
        _vw = self.resid_dropout(_vw)
        return _vw


class MLP(nn.Module):
    def __init__(self, indim, dim, activation=None, dropout=0.):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        self.projA = Conv1D(indim, dim)
        self.projB = Conv1D(dim, indim)
        self.act = ACT_FNS[activation]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.projA(x))
        h2 = self.projB(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, indim, bidir=False, numheads=None, activation="relu",
                 attention_dropout=0., residual_dropout=0., scale=False):
        super(Block, self).__init__()
        self.attn = MultiHeadAttention(indim, bidir=bidir, numheads=numheads,
           attention_dropout=attention_dropout, residual_dropout=residual_dropout,
           scale=scale)
        self.ln_1 = LayerNorm(indim)
        self.mlp = MLP(indim, 4 * indim, activation=activation, dropout=residual_dropout)
        self.ln_2 = LayerNorm(indim)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.float().unsqueeze(-1)
        a = self.attn(x, mask=mask)
        n = self.ln_1(x + a, mask=mask)
        m = self.mlp(n)
        h = self.ln_2(n + m, mask=mask)
        return h


class Transformer(nn.Module):
    """ Transformer model """

    def __init__(self, dim=512, worddic=None, numlayers=1, numheads=None, activation="relu",
                 bidir=False, embedding_dropout=0., attention_dropout=0., residual_dropout=0., scale=False):
        super(Transformer, self).__init__()
        self.embed = q.WordEmb(dim, worddic=worddic)
        self.drop = nn.Dropout(p=embedding_dropout)
        self.h = nn.ModuleList([Block(dim, bidir=bidir, numheads=numheads, activation=activation,
                                attention_dropout=attention_dropout, residual_dropout=residual_dropout,
                                scale=scale) for _ in range(numlayers)])

        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x, mask=None):
        x = x.view(-1, x.size(-2), x.size(-1))
        e, xmask = self.embed(x)
        mask = xmask & mask if (mask is not None and xmask is not None) else (mask if mask is not None else xmask)
        # Add the position information to the input embeddings
        h = e.sum(dim=2)
        for block in self.h:
            h = block(h, mask=mask)
        return h

# endregion