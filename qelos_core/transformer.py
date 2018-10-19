import math

import numpy as np
import torch
from torch import nn
from copy import copy

import qelos_core as q


# region from huggingface github transformer
class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, dim, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))
        self.e = e

    def forward(self, x):
        u = torch.mean(x, -1, keepdim=True)
        s = torch.mean((x - u).pow(2), -1, keepdim=True)
        x = (x - u) / (torch.sqrt(s + self.e))
        return self.g * x + self.b


class MultiHeadAttention(nn.Module):
    def __init__(self, indim=None, kdim=None, vdim=None, bidir=True, numheads=None,
                 attention_dropout=0., residual_dropout=0., scale=True, **kw):
        super(MultiHeadAttention, self).__init__(**kw)

        self.numheads, self.indim = numheads, indim
        self.bidir, self.scale = bidir, scale
        vdim = indim if vdim is None else vdim
        kdim = indim if kdim is None else kdim

        self.d_k = kdim // numheads     # dim per head in key and query
        self.d_v = vdim // numheads

        self.q_proj = nn.Linear(indim, numheads * self.d_k)
        self.k_proj = nn.Linear(indim, numheads * self.d_k)
        self.v_proj = nn.Linear(indim, numheads * self.d_v)
        nn.init.normal_(self.q_proj.weight, mean=0, std=np.sqrt(2.0 / (indim + self.d_k)))
        nn.init.normal_(self.k_proj.weight, mean=0, std=np.sqrt(2.0 / (indim + self.d_k)))
        nn.init.normal_(self.v_proj.weight, mean=0, std=np.sqrt(2.0 / (indim + self.d_v)))
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)

        self.vw_proj = nn.Linear(vdim, indim)
        nn.init.xavier_normal_(self.vw_proj.weight)
        nn.init.zeros_(self.vw_proj.bias)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(residual_dropout)

    def update_prev(self, k, v): return k, v

    def forward(self, x, k=None, v=None, mask=None):  # (batsize, <?>-seqlen, <?>-dim), mask on keys
        """
        :param x:   is input    (batsize, seqlen, indim)
        :param k:   if None, x is used for k proj, otherwise provided k
        :param v:   if None, k is used for v proj, otherwise provided v
        :param mask:    mask on keys (batsize, seqlen)
        :return:    (batsize, seqlen, indim)
        """
        batsize = x.size(0)
        q = x
        k = q if k is None else k
        v = k if v is None else v

        q = self.q_proj(q).view(batsize, q.size(1), self.numheads, self.d_k)
        k = self.k_proj(k).view(batsize, k.size(1), self.numheads, self.d_k)
        v = self.v_proj(v).view(batsize, v.size(1), self.numheads, self.d_v)

        k, v = self.update_prev(k, v)

        # compute attention weights
        w = torch.einsum("bshd,bzhd->bhsz", (q, k))     # (batsize, numheads, q_seqlen, k_seqlen)
        # scale attention weights
        if self.scale:
            w = w / math.sqrt(self.d_k)  # scale attention weights by dimension of keys

        # compute mask
        wholemask = None
        if mask is not None:
            # w = w + torch.log(mask.float().view(mask.size(0), 1, mask.size(1), 1))
            wholemask = mask.float().view(mask.size(0), 1, 1, mask.size(1))
        if self.bidir is False:
            seqlen = w.size(-1)
            causality_mask = torch.tril(torch.ones(seqlen, seqlen, device=x.device)).unsqueeze(0).unsqueeze(0)
                # .view(1, 1, seqlen, seqlen)
            wholemask = wholemask * causality_mask if wholemask is not None else causality_mask
            # * self.mask + -1e9 * (1 - self.mask)  # TF implem method: mask_attn_weights
        # apply mask on attention weights
        if wholemask is not None:
            w = w + torch.log(wholemask)

        # normalize and dropout attention weights
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # compute summaries based on attention weights w and values v
        vw = torch.einsum("bhsz,bzhd->bshd", (w, v))  # (batsize, seqlen, numheads, dim_per_head)
        ret_vw = vw
        # compute output
        new_shape = vw.size()[:-2] + (vw.size(-2) * vw.size(-1),)
        vw = vw.contiguous().view(*new_shape)
        _vw = self.vw_proj(vw)
        _vw = self.resid_dropout(_vw)
        return _vw #, torch.cat([_i.view(_i.size(0), _i.size(1), _i.size(2)*_i.size(3)) for _i in [q, k, v]], 2), \
               # ret_vw.transpose(2, 1)


class MultiHeadAttentionCell(nn.Module):
    """
    For incrementally predicting a next output given a single time step input and previous timestep values
    """
    def __init__(self, core:MultiHeadAttention, horizon:int=100, **kw):
        super(MultiHeadAttentionCell, self).__init__(**kw)
        self.core = copy(core)
        self.core.bidir = True
        self.core.update_prev = lambda k, v: self.update_prev(k, v)
        assert(core.bidir is False)     # ensure it was trained in decoder mode
        self._horizon = horizon
        self._prev_k = None      # (batsize, seqlen, numheads, dim)
        self._prev_v = None

    def rec_reset(self):
        self._prev_k = None
        self._prev_v = None

    def update_prev(self, k, v):
        """
        :param k:   (batsize, 1, numheads, dim_per_head)
        :param v:   (batsize, 1, numheads, dim_per_head)
        :return:
        """
        if self._prev_k is None:
            assert(self._prev_v is None)
            self._prev_k, self._prev_v = k, v
        else:
            self._prev_k = torch.cat([self._prev_k, k], 1)
            self._prev_v = torch.cat([self._prev_v, v], 1)
        assert(self._prev_k.size()[:-1] == self._prev_v.size()[:-1])
        if self._prev_k.size(1) > self._horizon:
            self._prev_k = self._prev_k[:, -self._horizon:]
            self._prev_v = self._prev_v[:, -self._horizon:]
        return self._prev_k, self._prev_v

    def forward(self, x, k=None, v=None, mask=None):    # TODO: also accumulate mask
        """
        :param x:       (batsize, 1, indim)
        :param k:       (batsize, 1, kdim)
        :param v:       (batsize, 1, vdim)
        :return:        (batsize, 1, indim)
        """
        if mask is not None:
            raise NotImplemented("TODO: implement mask accumulation in MultiHeadAttention and its cell")
        ret = self.core(x, k=k, v=v, mask=mask)
        return ret


class MLP(nn.Module):
    def __init__(self, indim, dim, activation=nn.ReLU, dropout=0.):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        self.projA = nn.Linear(indim, dim)
        self.projB = nn.Linear(dim, indim)
        nn.init.normal_(self.projA.weight, mean=0, std=np.sqrt(2.0 / (indim + dim)))
        nn.init.normal_(self.projB.weight, mean=0, std=np.sqrt(2.0 / (indim + dim)))
        nn.init.zeros_(self.projA.bias)
        nn.init.zeros_(self.projB.bias)
        self.act = activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.projA(x))
        h2 = self.projB(h)
        return self.dropout(h2)


class EncoderBlock(nn.Module):
    """ Normal self-attention block. Used in encoders. """
    def __init__(self, indim, kdim=None, vdim=None, numheads=None, activation=nn.ReLU,
                 attention_dropout=0., residual_dropout=0., scale=True, _bidir=True, **kw):
        """
        :param indim:       dimension of the input vectors
        :param kdim:        total dimension for the query and key projections
        :param vdim:        total dimension for the value projection
        :param bidir:       whether to run this in bidirectional (default) or uni-directional mode.
                            if uni-directional, this becomes a left-to-right LM-usable block by using triu mask
        :param numheads:    number of self-attention heads
        :param activation:  activation function to use
        :param attention_dropout:   dropout on attention
        :param residual_dropout:    dropout on residual
        :param scale:       whether to scale attention weights by dimension of value vectors
        :param kw:
        """
        super(EncoderBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(indim, kdim=kdim, vdim=vdim, bidir=_bidir, numheads=numheads,
           attention_dropout=attention_dropout, residual_dropout=residual_dropout, scale=scale)
        self.ln_slf = nn.LayerNorm(indim)
        self.mlp = MLP(indim, 4 * indim, activation=activation, dropout=residual_dropout)
        self.ln_ff = nn.LayerNorm(indim)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.float().unsqueeze(-1)
        # a = self.slf_attn(x, mask=mask)
        # h = self.mlp(a+x) + x
        #
        a = self.slf_attn(x, mask=mask)
        n = self.ln_slf(x + a)
        m = self.mlp(n)
        h = self.ln_ff(n + m)
        # h = m + n
        return h


class DecoderBlock(EncoderBlock):
    def __init__(self, indim, kdim=None, vdim=None, numheads=None, activation=nn.ReLU,
                 attention_dropout=0., residual_dropout=0., scale=True, noctx=False, **kw):
        super(DecoderBlock, self).__init__(indim, kdim=kdim, vdim=vdim, _bidir=False, numheads=numheads,
                activation=activation, attention_dropout=attention_dropout, residual_dropout=residual_dropout,
                scale=scale, **kw)
        # additional modules for attention to ctx
        self.noctx = noctx
        if not noctx:
            self.ctx_attn = MultiHeadAttention(indim, kdim=kdim, vdim=vdim, bidir=True, numheads=numheads,
               attention_dropout=attention_dropout, residual_dropout=residual_dropout, scale=scale)
            self.ln_ctx = LayerNorm(indim)

    def forward(self, x, ctx, mask=None, ctxmask=None):
        """
        :param x:       decoder input sequence of vectors   (batsize, seqlen_dec, dim)
        :param ctx:     encoded sequence of vectors         (batsize, seqlen_enc, dim)
        :param mask:    mask on the dec sequence   (batsize, seqlen_dec)
        :param ctxmask:    mask on the ctx (instead of mask on x) !!!     (batsize, seqlen_enc)
        :return:
        """
        if mask is not None:
            x = x * mask.float().unsqueeze(-1)
        # if ctxmask is not None:
        #     ctx = ctx * ctxmask.float().unsqueeze(-1)     # do we need this? no
        # self attention
        a = self.slf_attn(x, mask=mask)
        na = self.ln_slf(x + a)
        if self.noctx is False:
            # ctx attention
            b = self.ctx_attn(na, k=ctx, mask=ctxmask)
            nb = self.ln_ctx(na + b)
        else:   # skip the context part
            nb = na
        # ff
        m = self.mlp(nb)
        h = self.ln_ff(nb + m)
        return h


class DecoderBlockCell(nn.Module):
    def __init__(self, core:DecoderBlock, horizon:int=100, **kw):
        super(DecoderBlockCell, self).__init__(**kw)
        self.core = copy(core)
        self.horizon = horizon
        self.core.slf_attn = MultiHeadAttentionCell(self.core.slf_attn, horizon=horizon)

    def forward(self, x, ctx, mask=None, ctxmask=None):
        """
        :param x:       (batsize, 1, dim)
        :param ctx:     (batsize, seqlen, dim)
        :param ctxmask: (batsize, seqlen)
        :return:
        """
        ret = self.core(x, ctx, mask=mask, ctxmask=ctxmask)
        return ret


class TransformerEncoder(nn.Module):
    def __init__(self, dim=512, kdim=None, vdim=None, maxlen=512, numlayers=6, numheads=8, activation=nn.ReLU,
                 embedding_dropout=0., attention_dropout=0., residual_dropout=0., scale=True, **kw):
        super(TransformerEncoder, self).__init__(**kw)
        self.maxlen = maxlen
        posembD = {str(k): k for k in range(maxlen)}
        self.posemb = q.WordEmb(dim, worddic=posembD) if maxlen > -1 else None
        self.embdrop = nn.Dropout(p=embedding_dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(dim, kdim=kdim, vdim=vdim, numheads=numheads, activation=activation,
                         attention_dropout=attention_dropout, residual_dropout=residual_dropout, scale=scale)
            for _ in range(numlayers)
        ])

    def forward(self, x, mask=None):
        """
        :param x:       (batsize, seqlen, dim)
        :param mask:    optional mask (batsize, seqlen)
        :return:        (batsize, seqlen, outdim)
        """
        x = self.embdrop(x)     # TODO: or after adding position embeddings?

        emb = x
        if self.posemb is not None:
            assert(x.size(1) < self.maxlen)
            xpos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
            posemb, *_ = self.posemb(xpos)
            emb = x + posemb

        h = emb

        for layer in self.layers:
            h = layer(h, mask=mask)
        return h


class TransformerDecoder(TransformerEncoder):
    def __init__(self, dim=512, kdim=None, vdim=None, maxlen=512, numlayers=6, numheads=8, activation=nn.ReLU,
                 embedding_dropout=0., attention_dropout=0., residual_dropout=0., scale=True, noctx=False, **kw):
        super(TransformerDecoder, self).__init__(**kw)
        self.maxlen = maxlen
        self.noctx = noctx
        posembD = {str(k): k for k in range(maxlen)}
        self.posemb = q.WordEmb(dim, worddic=posembD) if maxlen > -1 else None
        self.embdrop = nn.Dropout(p=embedding_dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(dim, kdim=kdim, vdim=vdim, numheads=numheads, activation=activation,
                         attention_dropout=attention_dropout, residual_dropout=residual_dropout,
                         scale=scale, noctx=noctx)
            for _ in range(numlayers)
        ])

    def forward(self, x, ctx, mask=None, ctxmask=None, _posoffset:int=0):
        """
        :param x:       same is Encoder
        :param ctx:     (batsize, seqlen_ctx, encdim)
        :param mask:    (batsize, seqlen_out)
        :param ctxmask:     (batsize, seqlen_ctx)
        :return:
        """
        x = self.embdrop(x)       # TODO: or after adding position embeddings?

        emb = x
        if self.posemb is not None:
            assert(x.size(1) <= self.maxlen - _posoffset)
            xpos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0) + _posoffset
            posemb, *_ = self.posemb(xpos)
            emb = emb + posemb

        h = emb

        for layer in self.layers:
            h = layer(h, ctx, mask=mask, ctxmask=ctxmask)
        return h


class TransformerDecoderCell(nn.Module):
    def __init__(self, core:TransformerDecoder, horizon:int=100, **kw):
        super(TransformerDecoderCell, self).__init__(**kw)
        self.core = copy(core)
        self.horizon = horizon
        self.core.layers = nn.ModuleList([
            DecoderBlockCell(decoderblock, horizon=horizon)
            for decoderblock in self.core.layers
        ])
        self._posoffset = 0

        # TODO: finish: copy necesary attributes

    def rec_reset(self):
        self._posoffset = 0

    def forward(self, x, ctx, mask=None, ctxmask=None):
        """
        :param x:       (batsize, 1, dim)
        :param ctx:     (batsize, seqlen, dim)
        :param ctxmask: (batsize, seqlen)
        :return:
        """
        ret = self.core(x, ctx, mask=None, ctxmask=None, _posoffset=self._posoffset)
        self._posoffset += 1
        return ret


class TS2S(nn.Module):
    def __init__(self, encoder:TransformerEncoder, decoder:TransformerDecoder, **kw):
        super(TS2S, self).__init__(**kw)
        self.encoder, self.decoder = encoder, decoder

    def forward(self, x, y, xmask=None, ymask=None):
        """
        :param x:       (batsize, inpseqlen)
        :param y:       (batsize, outseqlen)
        :return:
        """
        xemb = self.encoder(x, mask=xmask)
        out = self.decoder(y, xemb, mask=ymask, ctxmask=xmask)
        return out


class TS2SCell(nn.Module):
    def __init__(self, core:TS2S, horizon:int=100, **kw):
        super(TS2SCell, self).__init__(**kw)
        self.core = core
        self.horizon = horizon
        self.encoder = core.encoder
        self.decoder = TransformerDecoderCell(core.decoder, horizon=horizon)
        self._x = None
        self._ctx = None
        self._ctxmask = None

    def rec_reset(self):
        self._x, self._ctx, self._ctxmask = None, None, None

    def forward(self, x, y, xmask=None, ymask=None):
        """
        :param x:       (batsize, seqlen_enc)
        :param y:       (batsize, 1)
        :param xmask:
        :param ymask:
        :return:
        """
        if self._ctx is None:
            self._ctx = self.encoder(x, mask=xmask)
            self._ctxmask = xmask
        else:
            pass
            # assert(x == self._x)
            # assert(xmask == self._ctxmask)

        out = self.decoder(y, self._ctx, mask=ymask, ctxmask=self._ctxmask)
        return out


class TS2S_arg(TS2S):
    def __init__(self, dim=512, kdim=None, vdim=None, maxlen=512, numlayers=6, numheads=8,
                 activation=nn.ReLU, embedding_dropout=0., attention_dropout=0., residual_dropout=0.,
                 scale=True, **kw):
        encoder = TransformerEncoder(dim=dim, kdim=kdim, vdim=vdim, maxlen=maxlen, numlayers=numlayers,
                                     numheads=numheads, activation=activation,
                                     embedding_dropout=embedding_dropout, attention_dropout=attention_dropout,
                                     residual_dropout=residual_dropout, scale=scale)
        decoder = TransformerDecoder(dim=dim, kdim=kdim, vdim=vdim, maxlen=maxlen, numlayers=numlayers,
                                     numheads=numheads, activation=activation,
                                     embedding_dropout=embedding_dropout, attention_dropout=attention_dropout,
                                     residual_dropout=residual_dropout, scale=scale, noctx=False)
        super(TS2S_arg, self).__init__(encoder, decoder, **kw)

# endregion