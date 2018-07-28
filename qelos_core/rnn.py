import torch
import qelos_core as q
from qelos_core.basic import Dropout
import numpy as np
import re


def rec_reset(module):
    """ resets the rec states (incl. RNN states, dropouts etc) of the module and all its descendants
        by calling their rec_reset(), if present() """
    for modu in module.modules():
        if hasattr(modu, "rec_reset"):
            modu.rec_reset()


class RecDropout(Dropout):
    """ Variational Dropout """
    def __init__(self, p=.5):
        super(RecDropout, self).__init__(p=p)
        self.mask = None

    def rec_reset(self):
        self.mask = None

    def forward(self, *x):
        y = x
        if self.training:
            if self.mask is None:
                self.mask = map(lambda xe: self.d(torch.ones_like(xe).to(xe.device)), x)
            y = map(lambda (xe, me): xe * me, zip(x, self.mask))
        y = y[0] if len(y) == 1 else y
        return y


class Zoneout(RecDropout):
    def forward(self, *x):
        y = [xe[1] for xe in x]
        if self.training:
            if self.mask is None:
                self.mask = map(lambda (_, xe): self.d(torch.ones_like(xe).to(xe.device)).clamp(0, 1), x)
            y = [(1 - zoner) + h_t + zoner * h_tm1
                 for (h_tm1, h_t), zoner in zip(x, self.mask)]
        y = y[0] if len(y) == 1 else y
        return y


class PositionwiseForward(torch.nn.Module):       # TODO: make Recurrent
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, activation="relu", dropout=0.1):
        super(PositionwiseForward, self).__init__()
        self.w_1 = torch.nn.Conv1d(d_hid, d_inner_hid, 1) # position-wise
        self.w_2 = torch.nn.Conv1d(d_inner_hid, d_hid, 1) # position-wise
        self.layer_norm = q.LayerNormalization(d_hid)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation_fn = q.name2fn(activation)()

    def forward(self, x):
        residual = x
        output = self.activation_fn(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class RecCell(torch.nn.Module):
    def __init__(self, dropout_in=None, dropout_rec=None, zoneout=None, **kw):
        super(RecCell, self).__init__(**kw)
        self.dropout_in, self.dropout_rec, self.zoneout = dropout_in, dropout_rec, zoneout

        if self.dropout_in and not isinstance(self.dropout_in, q.Dropout):
            self.dropout_in = RecDropout(p=self.dropout_in)
        if self.dropout_rec and not isinstance(self.dropout_rec, q.Dropout):
            self.dropout_rec = RecDropout(p=self.dropout_rec)
        if self.zoneout and not isinstance(self.zoneout, (Zoneout,)):
            self.zoneout = Zoneout(p=self.zoneout)
        assert(isinstance(self.zoneout, (Zoneout, type(None))))
        assert(isinstance(self.zoneout, (q.Dropout, type(None))))

    def rec_reset(self):
        if isinstance(self.dropout_in, RecDropout):
            self.dropout_in.rec_reset()
        if isinstance(self.dropout_rec, RecDropout):
            self.dropout_rec.rec_reset()
        if isinstance(self.zoneout, Zoneout):
            self.zoneout.rec_reset()

    def apply_mask_t(self, *statepairs, **kw):
        """ interpolates between previous and new state inside a timestep in a batch based on mask"""
        mask_t = q.getkw(kw, "mask_t", None)
        if mask_t is not None:
            mask_t = mask_t.float().unsqueeze(1)
            ret = [h_t * mask_t + h_tm1 * (1 - mask_t) for h_tm1, h_t in statepairs]
            return tuple(ret)
        else:
            return tuple([statepair[1] for statepair in statepairs])


class GRUCell(RecCell):
    """ wrapper around PyTorch GRUCell with extra features """
    def __init__(self, indim, outdim, bias=True,
                 dropout_in=None, dropout_rec=None, zoneout=None, **kw):
        super(GRUCell, self).__init__(dropout_in=dropout_in, dropout_rec=dropout_rec, zoneout=zoneout, **kw)
        self.indim, self.outdim, self.bias, = indim, outdim, bias

        self.cell = torch.nn.GRUCell(self.indim, self.outdim, bias=self.bias)

        self.h_tm1 = None
        self.h_0 = q.val(torch.zeros(1, outdim)).v

    def rec_reset(self):
        self.h_tm1 = None      # reset state
        super(GRUCell, self).rec_reset()

    def forward(self, x_t, mask_t=None, **kw):
        batsize = x_t.size(0)
        x_t = self.dropout_in(x_t) if self.dropout_in else x_t

        # previous state
        h_tm1 = self.h_0.expand(batsize, -1) if self.h_tm1 is None else self.h_tm1
        h_tm1 = self.dropout_rec(h_tm1) if self.dropout_rec else h_tm1

        h_t = self.cell(x_t, h_tm1)

        # next state
        h_t = self.zoneout((h_tm1, h_t)) if self.zoneout else h_t
        h_t, = self.apply_mask_t((h_tm1, h_t), mask_t=mask_t)
        self.h_tm1 = h_t
        return h_t


class LSTMCell(RecCell):
    """ wrapper around PyTorch GRUCell with extra features """
    def __init__(self, indim, outdim, bias=True,
                 dropout_in=None, dropout_rec=None, zoneout=None, **kw):
        super(LSTMCell, self).__init__(dropout_in=dropout_in, dropout_rec=dropout_rec, zoneout=zoneout, **kw)
        self.indim, self.outdim, self.bias = indim, outdim, bias

        self.cell = torch.nn.LSTMCell(self.indim, self.outdim, bias=self.bias)

        self.y_tm1 = None
        self.c_tm1 = None
        self.y_0 = q.val(torch.zeros(1, outdim)).v
        self.c_0 = q.val(torch.zeros(1, outdim)).v

    def rec_reset(self):
        self.y_tm1 = None
        self.c_tm1 = None
        super(LSTMCell, self).rec_reset()

    def forward(self, x_t, mask_t=None, **kw):
        batsize = x_t.size(0)
        x_t = self.dropout_in(x_t) if self.dropout_in else x_t

        # previous states
        y_tm1 = self.y_0.expand(batsize, -1) if self.y_tm1 is None else self.y_tm1
        c_tm1 = self.c_0.expand(batsize, -1) if self.c_tm1 is None else self.c_tm1
        y_tm1, c_tm1 = self.dropout_rec(y_tm1, c_tm1) if self.dropout_rec else (y_tm1, c_tm1)

        y_t, c_t = self.cell(x_t, (y_tm1, c_tm1))

        # next state
        y_t, c_t = self.zoneout((y_tm1, y_t), (c_tm1, c_t)) if self.zoneout else (y_t, c_t)
        y_t, c_t = self.apply_mask_t((y_tm1, y_t), (c_tm1, c_t), mask_t=mask_t)
        self.y_tm1, self.c_tm1 = y_t, c_t
        return y_t


class AttentionBase(torch.nn.Module):
    def __init__(self, **kw):
        super(AttentionBase, self).__init__()
        self.sm = torch.nn.Softmax(-1)

    def forward(self, q, ctx, ctx_mask=None, values=None):
        return self._forward(q, ctx, ctx_mask=ctx_mask, values=values)


class Attention(AttentionBase):
    def __init__(self, **kw):
        super(Attention, self).__init__(**kw)
        self.alpha_tm1, self.summ_tm1 = None, None

    def forward(self, q, ctx, ctx_mask=None, values=None):
        """
        :param ctx:     context (keys), (batsize, seqlen, dim)
        :param q:       query, (batsize, dim)
        :param ctxmask: context mask (batsize, seqlen)
        :param values:  values to summarize, (batsize, seqlen, dim), if unspecified, ctx is used
        :return:        attention alphas (batsize, seqlen) and summary (batsize, dim)
        """
        alphas, summary, scores = super(Attention, self).forward(q, ctx, ctx_mask=None, values=values)
        self.alpha_tm1, self.summ_tm1 = alphas, summary
        return alphas, summary, scores

    def rec_reset(self):
        self.alpha_tm1, self.summ_tm1 = None, None


class AttentionWithCoverage(Attention):
    def __init__(self):
        super(AttentionWithCoverage, self).__init__()
        self.coverage = None
        self.cov_count = 0
        self._cached_ctx = None
        self.penalty = None #AttentionCoveragePenalty()

    def reset_coverage(self):
        self.coverage = None
        self.cov_count = 0
        self._cached_ctx = None

    def rec_reset(self):
        super(AttentionWithCoverage, self).rec_reset()
        self.reset_coverage()

    def forward(self, q, ctx, ctx_mask=None, values=None):
        alphas, summary, scores = super(AttentionWithCoverage, self).forward(q, ctx, ctx_mask=ctx_mask, values=values)
        self.update_penalties(alphas)
        self.update_coverage(alphas, ctx)
        return alphas, summary, scores

    def update_penalties(self, alphas):
        if self.coverage is not None:
            overlap = torch.min(self.coverage, alphas).sum(1)
            self.penalty(overlap)

    def update_coverage(self, alphas, ctx):
        if self.coverage is None:
            self.coverage = torch.zeros_like(alphas)
            self._cached_ctx = ctx
        assert((self._cached_ctx - ctx).norm() < 1e-5)
        self.coverage += alphas
        self.cov_count += 1


class AttentionWithMonotonicCoverage(AttentionWithCoverage):
    def __init__(self):
        super(AttentionWithMonotonicCoverage, self).__init__()

    def update_coverage(self, alphas, ctx):
        if self.coverage is None:
            self.coverage = torch.zeros_like(alphas)
            self._cached_ctx = ctx
        assert ((self._cached_ctx - ctx).norm() < 1e-5)

        cum_alphas = 1 - torch.cumsum(alphas, 1)
        cum_alphas = torch.cat([cum_alphas[:, 0:1], cum_alphas[:, :-1]], 1)
        self.coverage += cum_alphas
        self.cov_count += 1


class _DotAttention(AttentionBase):
    def _forward(self, q, ctx, ctx_mask=None, values=None):
        scores = torch.bmm(ctx, q.unsqueeze(2)).squeeze(2)
        scores = scores + (torch.log(ctx_mask.float()) if ctx_mask is not None else 0)
        alphas = self.sm(scores)
        values = ctx if values is None else values
        summary = values * alphas.unsqueeze(2)
        summary = summary.sum(1)
        return alphas, summary, scores


class DotAttention(Attention, _DotAttention):
    pass


class _GeneralDotAttention(AttentionBase):
    def __init__(self, ctxdim=None, qdim=None, **kw):
        super(_GeneralDotAttention, self).__init__(**kw)
        self.W = torch.nn.Parameter(torch.empty(ctxdim, qdim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.W)

    def _forward(self, q, ctx, ctx_mask=None, values=None):
        projq = torch.matmul(self.W, q)     # (batsize, ctxdim)
        alphas, summary, scores = super(_GeneralDotAttention, self)._forward(projq, ctx, ctx_mask=ctx_mask, values=values)
        return alphas, summary, scores


class GeneralDotAttention(Attention, _GeneralDotAttention):
    pass


class _FwdAttention(AttentionBase):
    def __init__(self, ctxdim=None, qdim=None, attdim=None, nonlin=torch.nn.Tanh(), **kw):
        super(_FwdAttention, self).__init__(**kw)
        self.linear = torch.nn.Linear(ctxdim + qdim, attdim)
        self.nonlin = nonlin
        self.afterlinear = torch.nn.Linear(attdim, 1)

    def _forward(self, q, ctx, ctx_mask=None, values=None):
        q = q.unsqueeze(1).repeat(1, ctx.size(1), 1)
        x = torch.cat([ctx, q], 2)
        y = self.linear(x)      # (batsize, seqlen, attdim)
        scores = self.afterlinear(y).squeeze(2)
        scores = scores + (torch.log(ctx_mask.float()) if ctx_mask is not None else 0)
        alphas = self.sm(scores)
        values = ctx if values is None else values
        summary = values * alphas.unsqueeze(2)
        summary = summary.sum(1)
        return alphas, summary, scores


class FwdAttention(Attention, _FwdAttention):
    pass


class _FwdMulAttention(AttentionBase):
    def __init__(self, indim=None, attdim=None, nonlin=torch.nn.Tanh(), **kw):
        super(_FwdMulAttention, self).__init__(**kw)
        self.linear = torch.nn.Linear(indim * 3, attdim)
        self.nonlin = nonlin
        self.afterlinear = torch.nn.Linear(attdim, 1)

    def _forward(self, q, ctx, ctx_mask=None, values=None):
        q = q.unsqueeze(1).repeat(1, ctx.size(1), 1)
        x = torch.cat([ctx, q, ctx * q], 2)
        y = self.linear(x)      # (batsize, seqlen, attdim)
        scores = self.afterlinear(y).squeeze(2)
        scores = scores + (torch.log(ctx_mask.float()) if ctx_mask is not None else 0)
        alphas = self.sm(scores)
        values = ctx if values is None else values
        summary = values * alphas.unsqueeze(2)
        summary = summary.sum(1)
        return alphas, summary, scores


class FwdMulAttention(Attention, _FwdMulAttention):
    pass


class RelationContentAttention(Attention):
    """
    Defines a composite relation-content attention.
    - Encoded sequence must start with a special token shared across all examples (e.g. always start with <START>.
    - in every batch, before using this attention, rel_map must be set using set_rel_map()
    """

    def __init__(self, relemb=None, query_proc=None, **kw):
        """
        :param relemb:      embedder for relation ids
        :param query_proc:  (optional) module that returns query to use against content
                            and query to use against relation.
                            Default: take original query against concat of content and relation ctx
        :param kw:
        """
        super(RelationContentAttention, self).__init__(**kw)
        self.rel_map = None  # rel maps for current batch (batsize, seqlen, seqlen) integer ids of relations
        self.relemb = relemb
        self._rel_map_emb = None  # cached augmented ctx -- assumed that ctx is not changed between time step
        self.query_proc = query_proc

    def rec_reset(self):
        super(RelationContentAttention, self).rec_reset()
        self.rel_map = None
        self._rel_map_emb = None

    def set_rel_map(self, relmap):  # sets relation map and embeds
        """
        :param relmap:  (batsize, seqlen, seqlen) integers with rel ids
        :return:
        """
        self.rel_map = relmap
        rel_map_emb = self.relemb(relmap)  # (batsize, seqlen, seqlen, relembdim)
        if isinstance(rel_map_emb, tuple) and len(rel_map_emb) == 2:
            rel_map_emb, _ = rel_map_emb
        self._rel_map_emb = rel_map_emb

    def get_rel_ctx(self, ctx):
        if self.alpha_tm1 is None:  # in first time step, alpha_tm1 is assumed to have been on first element of encoding
            alphas_tm1 = torch.zeros_like(self._rel_map_emb[:, :, 0, 0])
            alphas_tm1[:, 0] = 1.
        else:
            alphas_tm1 = self.alphas_tm1  # (batsize, seqlen)

        alphas_tm1 = alphas_tm1.unsqueeze(2).unsqueeze(2)  # (batsize, seqlen, 1, 1)
        rel_summ = alphas_tm1 * self._rel_map_emb
        rel_summ = rel_summ.sum(2)  # (batsize, seqlen, relembdim)

        return rel_summ

    def forward(self, q, ctx, ctx_mask=None, values=None):
        rel_ctx = self.get_rel_ctx(ctx)
        aug_ctx = torch.cat([ctx, rel_ctx], 2)
        if self.query_proc is not None:
            cont_q, rel_q = self.query_proc(q)
            q = torch.cat([cont_q, rel_q], 1)
        return super(RelationContentAttention, self).forward(q, aug_ctx, ctx_mask=ctx_mask, values=values)


class RelationContextAttentionSeparated(RelationContentAttention):
    """
    Similar as RelationContentAttention,
    but with explicit prediction of probability of doing content-based vs relation-based attention.
    Choices:
    - shared decoder / separated decoder
    - query_proc:  module that given query vector, produces tuple (content_query, relation_query, cont_vs_rel_prob)
    """
    def __init__(self, rel_att=None, relemb=None, query_proc=None, **kw):
        super(RelationContextAttentionSeparated, self).__init__(relemb=relemb, query_proc=query_proc, **kw)
        self.rel_att = rel_att

    def forward(self, q, ctx, ctx_mask=None, values=None):
        cont_q, rel_q, vs_prob = self.query_proc(q)
        rel_ctx = self.get_rel_ctx(ctx)
        rel_alphas, rel_summaries, rel_scores = self.rel_att(rel_q, rel_ctx, ctx_mask=ctx_mask)
        cont_alphas, cont_summaries, cont_scores = \
            super(RelationContextAttentionSeparated, self)\
            .forward(cont_q, ctx, ctx_mask=ctx_mask, values=values)
        vs_prob = vs_prob.unsqueeze(1)
        alphas = rel_alphas * vs_prob + cont_alphas * (1 - vs_prob)
        values = ctx if values is None else values
        summary = values * alphas.unsqueeze(2)
        summary = summary.sum(1)
        return alphas, summary


class Decoder(torch.nn.Module):
    """ abstract decoder """
    def __init__(self, cell, **kw):
        """
        :param cell:    must produce probabilities as first output
        :param kw:
        """
        super(Decoder, self).__init__(**kw)
        self.cell = cell

    def forward(self, xs, **kw):
        """
        :param xs:      argument(s) that will be time-sliced
        :param kw:      are passed to decoder cell (unless used in decoder itself)
        :return:
        """
        raise NotImplemented("use subclass")


class ThinDecoder(Decoder):
    """
    Thin decoder, cells have full control and decoder only provides time steps and merges outputs.
    Cell must implement:
        - forward(t, *args, **kw) -- cell must save all necessary outputs by itself, inputs are forwarded from decoder
        - stop() to terminate decoding
    """
    def forward(self, *args, **kw):
        q.rec_reset(self.cell)
        outs = []
        out_is_seq = False
        for t in xrange(10e9):
            outs_t = self.cell(t, *args, **kw)
            if q.issequence(outs_t):
                out_is_seq = True
            else:
                outs_t = (outs_t,)
            outs.append(outs_t)
            if self.cell.stop():
                break
        outs = zip(*outs)
        outs = tuple([torch.cat([a_i.unsqueeze(1) for a_i in a], 1) for a in outs])
        outs = outs[0] if not out_is_seq else outs
        return outs


class TFDecoder(Decoder):
    def forward(self, xs, **kw):
        q.rec_reset(self.cell)
        x_is_seq = True
        if not q.issequence(xs):
            x_is_seq = False
            xs = (xs,)
        outs = []
        out_is_seq = False
        for t in range(xs[0].size(1)):
            x_ts = tuple([x[:, t] for x in xs])
            x_ts = x_ts[0] if not x_is_seq else x_ts
            outs_t = self.cell(x_ts, **kw)
            if q.issequence(outs_t):
                out_is_seq = True
            else:
                outs_t = (outs_t,)
            outs.append(outs_t)
        outs = zip(*outs)
        outs = tuple([torch.cat([a_i.unsqueeze(1) for a_i in a], 1) for a in outs])
        outs = outs[0] if not out_is_seq else outs
        return outs


class FreeDecoder(Decoder):
    def __init__(self, cell, maxtime=None, **kw):
        super(FreeDecoder, self).__init__(cell, **kw)
        self.maxtime = maxtime

    def forward(self, xs, maxtime=None, **kw):
        """
        :param xs:
        :param maxtime:
        :param kw:      are passed directly into cell at every time step
        :return:
        """
        q.rec_reset(self.cell)
        maxtime = maxtime if maxtime is not None else self.maxtime
        x_is_seq = True
        if not q.issequence(xs):
            x_is_seq = False
            xs = (xs,)
        outs = []
        out_is_seq = False
        for t in range(maxtime):
            if t == 0:      # first time step --> use xs
                x_ts = xs
                x_ts = x_ts[0] if not x_is_seq else x_ts
            outs_t = self.cell(x_ts, **kw)
            x_ts = self._get_xs_from_ys(outs_t)      # --> get inputs from previous outputs
            if q.issequence(outs_t):
                out_is_seq = True
            else:
                outs_t = (outs_t,)
            outs.append(outs_t)
        outs = zip(*outs)
        outs = tuple([torch.cat([a_i.unsqueeze(1) for a_i in a], 1) for a in outs])
        outs = outs[0] if not out_is_seq else outs
        return outs

    def _get_xs_from_ys(self, ys):
        if hasattr(self.cell, "get_xs_from_ys"):
            xs = self.cell.get_xs_from_ys(ys)
        else:
            xs = self.get_xs_from_ys(ys)
        return xs

    def get_xs_from_ys(self, ys):       # default argmax implementation with out output
        assert(not q.issequence(ys))
        assert(ys.dim() == 2)   # (batsize, outsyms)
        _, argmax_ys = torch.max(ys, 1)
        xs = argmax_ys
        return xs


class DynamicOracleDecoder(Decoder):
    def __init__(self, cell, tracker=None, mode="sample", eps=0.2, explore=0., **kw):
        super(DynamicOracleDecoder, self).__init__(cell, **kw)
        self.mode = mode
        modere = re.compile("(\w+)-(\w+)")
        m = re.match(modere, mode)
        if m:
            self.gold_mode, self.next_mode = m.group(1), m.group(2)
            self.modes_split = True
        else:
            self.gold_mode, self.next_mode = mode, mode
            self.modes_split = False
        self.eps = eps
        self.explore = explore
        #
        self.tracker = tracker
        self.seqacc = []        # history of what has been fed to next time step
        self.goldacc = []       # use for supervision

        self._argmax_in_eval = True

    def rec_reset(self):
        self.reset()

    def reset(self):
        self.seqacc = []
        self.goldacc = []
        self.tracker.reset()

    def get_sequence(self):
        """ get the chosen output sequence """
        ret = torch.stack(self.seqacc, 1)
        return ret

    def get_gold_sequence(self):
        ret = torch.stack(self.goldacc, 1)
        return ret

    def forward(self, xs, maxtime=None, **kw):
        """
        :param xs:          tuple of (eids, ...) - eids being ids of the examples
                            --> eids not fed to decoder cell, everything else is, as usual
        :param maxtime:
        :param kw:          are passed directly into cell at every time step
        :return:
        """
        q.rec_reset(self.cell)
        assert(q.issequence(xs) and len(xs) >= 2)
        eids, xs = xs[0], xs[1:]
        maxtime = maxtime if maxtime is not None else self.maxtime
        x_is_seq = True
        if not q.issequence(xs):
            x_is_seq = False
            xs = (xs,)
        outs = []
        out_is_seq = False
        for t in range(maxtime):
            if t == 0:      # first time step --> use xs
                x_ts = xs
                x_ts = x_ts[0] if not x_is_seq else x_ts
            outs_t = self.cell(x_ts, **kw)
            x_ts, g_ts = self._get_xs_and_gs_from_ys(outs_t, eids)
            # --> get inputs for next time step and gold for current time step from previous outputs

            # store sampled
            self.seqacc.append(x_ts)
            self.goldacc.append(g_ts)

            if q.issequence(outs_t):
                out_is_seq = True
            else:
                outs_t = (outs_t,)
            outs.append(outs_t)
            if self.check_terminate():
                break
        outs = zip(*outs)
        outs = tuple([torch.cat([a_i.unsqueeze(1) for a_i in a], 1) for a in outs])
        outs = outs[0] if not out_is_seq else outs
        return outs

    def _get_xs_and_gs_from_ys(self, ys, eids):
        eids = eids.cpu().detach().numpy()
        if q.issequence(ys):
            assert(len(ys) == 1)
            y_t = ys[0]
        else:
            y_t = ys

        # compute prob mask
        ymask = torch.zeros_like(y_t)
        for i, eid in enumerate(eids):
            validnext = self.tracker.get_valid_next(eid)            # set of ids
            if isinstance(validnext, tuple) and len(validnext) == 2:
                raise q.SumTingWongException("no anvt supported")
            ymask[i, list(validnext)] = 1.

        # get probs
        _y_t = y_t + torch.log(ymask)
        goldprobs = self.sm(_y_t)

        assert(not self.training)

        if self.mode in "zerocost nocost".split():
            _, y_best = y_t.max(1)          # argmax from all
            _, gold_t = goldprobs.max(1)    # argmax from VNT
            y_random_valid = torch.distributions.Categorical(ymask).sample()        # uniformly sampled from VNT
            y_best_is_valid = torch.gather(ymask, 1, y_best.unsqueeze(1)).long()    # y_best is in VNT
            nextcat = torch.stack([y_random_valid, y_best], 1)
            x_t = torch.gather(nextcat, 1, y_best_is_valid).squeeze(1)              # if best is valid, takes best, else random valid
            if self.mode == "nocost":   # set mask as gold if best is valid --> no improvement if best is correct
                zero_gold = torch.zeros_like(gold_t).long()
                goldcat = torch.stack([gold_t, zero_gold], 1)
                gold_t = torch.gather(goldcat, 1, y_best_is_valid).squeeze(1)

        else:

            def _sample_using_mode(_goldprobs, _ymask, _mode):
                if _mode == "sample":
                    _ret_t = torch.distributions.Categorical(_goldprobs).sample()
                elif _mode == "uniform":
                    _ret_t = torch.distributions.Categorical(_ymask).sample()
                elif _mode == "esample":
                    _ret_t = torch.distributions.Categorical(_goldprobs).sample()
                    _alt_ret_t = torch.distributions.Categorical(_ymask).sample()
                    _epsprobs = (torch.rand_like(_ret_t) < self.eps).long()
                    _ret_t = torch.gather(torch.stack([_ret_t, _alt_ret_t], 1), 1, _epsprobs.unsqueeze(1)).squeeze(1)
                elif _mode == "argmax":
                    _, _ret_t = torch.max(_goldprobs, 1)
                else:
                    raise q.SumTingWongException("unsupported mode: {}".format(_mode))
                return _ret_t

            gold_t = _sample_using_mode(goldprobs, ymask, self.gold_mode)

            if self.explore == 0:
                if not self.modes_split:
                    x_t = gold_t
                else:
                    x_t = _sample_using_mode(goldprobs, ymask, self.next_mode)
            else:
                raise NotImplemented("exporing not supported")

        # update tracker
        for x_t_e, eid, gold_t_e in zip(x_t.cpu().detach().numpy(), eids, gold_t.cpu().detach().numpy()):
            self.tracker.update(eid, x_t_e, alt_x=gold_t_e)

        return x_t, gold_t


class BasicDecoderCell(torch.nn.Module):
    def __init__(self, emb, core, out):
        super(BasicDecoderCell, self).__init__()
        self.emb, self.core, self.out = emb, core, out

    def forward(self, x_t, ctx=None, **kw):
        """
        :param x_t:     tensor or list of tensors for this time step
        :param ctx:     inputs for all time steps, forwarded by decoder
        :param kw:      kwargs for all time steps, forwarded by decoder
        :return:
        """
        emb = self.emb(x_t)
        acts = self.core(emb)
        outs = self.out(acts)
        return outs


class LuongCell(torch.nn.Module):
    def __init__(self, emb=None, core=None, att=None, merge=None, out=None,
                 feed_att=False, return_alphas=False, return_scores=False, return_other=False, **kw):
        """

        :param emb:
        :param core:
        :param att:
        :param merge:
        :param out:         if None, out_vec (after merge) is returned
        :param feed_att:
        :param h_hat_0:
        :param kw:
        """
        super(LuongCell, self).__init__(**kw)
        self.emb, self.core, self.att, self.merge, self.out = emb, core, att, merge, out
        self.return_outvecs = self.out is None
        self.feed_att = feed_att
        self._h_hat_tm1 = None
        self.h_hat_0 = None
        self.return_alphas = return_alphas
        self.return_scores = return_scores
        self.return_other = return_other

    def rec_reset(self):
        self.h_hat_0 = None
        self._h_hat_tm1 = None

    def forward(self, x_t, ctx=None, ctx_mask=None, **kw):
        assert (ctx is not None)
        embs = self.emb(x_t)
        if q.issequence(embs) and len(embs) == 2:
            embs, mask = embs

        core_inp = embs
        if self.feed_att:
            if self._h_hat_tm1 is None:
                assert (self.h_hat_0 is not None)   #"h_hat_0 must be set when feed_att=True"
                self._h_hat_tm1 = self.h_hat_0
            core_inp = torch.cat([core_inp, self._h_hat_tm1], 1)
        core_out = self.core(core_inp)

        alphas, summaries, scores = self.att(core_out, ctx, ctx_mask=ctx_mask, values=ctx)

        out_vec = torch.cat([core_out, summaries], 1)
        out_vec = self.merge(out_vec) if self.merge is not None else out_vec
        self._h_hat_tm1 = out_vec

        ret = tuple()
        if not self.return_outvecs:
            out_vec = self.out(out_vec)
        ret += (out_vec,)

        if self.return_alphas:
            ret += (alphas,)
        if self.return_scores:
            ret += (scores,)
        if self.return_other:
            ret += (embs, core_out, summaries)
        return ret[0] if len(ret) == 1 else ret


class DecoderCell(LuongCell):
    pass


class OldDecoderCell(torch.nn.Module):
    def __init__(self, emb, core, att, out, feed_att=False, summ_0=None):
        print("WARNING: do not use this, use LuongCell instead")
        """
        :param emb:
        :param core:
        :param att:
        :param out:
        :param feed_att:    feed attention summary as input to next time step
        :param summ_0:
        """
        super(OldDecoderCell, self).__init__()
        self.emb, self.core, self.att, self.out = emb, core, att, out
        self.feed_att = feed_att
        self.summ_tm1 = summ_0
        assert(not feed_att or summ_0 is not None)  # "summ_0 must be specified if feeding attention summary to next time step"
        self.use_cell_out = True
        self.use_att_sum = True
        self.use_x_t_emb = False

    def forward(self, x_t, ctx=None, ctx_mask=None, **kw):
        assert(ctx is not None)
        embs = self.emb(x_t)
        if q.issequence(embs) and len(embs) == 2:
            embs, mask = embs

        core_inp = embs
        if self.feed_att:
            core_inp = torch.cat([core_inp, self.summ_tm1], 1)
        acts = self.core(core_inp)

        alphas, summaries = self.att(acts, ctx, ctx_mask=ctx_mask, values=ctx)
        self.summ_tm1 = summaries

        to_out = []
        if self.use_cell_out:
            to_out.append(acts)
        if self.use_att_sum:
            to_out.append(summaries)
        if self.use_x_t_emb:
            to_out.append(embs)
        to_out = torch.cat(to_out, 1)

        outscores = self.out(to_out)
        return outscores


class BahdanauCell(torch.nn.Module):
    """ Almost Bahdanau-style cell, except c_tm1 is fed as input to top decoder layer (core2),
            instead of as part of state """
    def __init__(self, emb=None, core1=None, core2=None, att=None, out=None,
                 return_alphas=False, return_scores=False, return_other=False, **kw):
        super(BahdanauCell, self).__init__(**kw)
        self.emb, self.core1, self.core2, self.att, self.out = emb, core1, core2, att, out
        self.return_outvecs = self.out is None
        self.summ_0 = None
        self._summ_tm1 = None
        self.return_alphas = return_alphas
        self.return_other = return_other
        self.return_scores = return_scores

    def rec_reset(self):
        self.summ_0 = None
        self._summ_tm1 = None

    def forward(self, x_t, ctx=None, ctx_mask=None, **kw):
        assert (ctx is not None)
        embs = self.emb(x_t)
        if q.issequence(embs) and len(embs) == 2:
            embs, mask = embs

        core_inp = embs
        core_out = self.core1(core_inp)

        if self._summ_tm1 is None:
            assert (self.summ_0 is not None)    # "summ_0 must be set"
            self._summ_tm1 = self.summ_0

        core_inp = torch.cat([core_out, self._summ_tm1], 1)
        core_out = self.core2(core_inp)

        alphas, summaries, scores = self.att(core_out, ctx, ctx_mask=ctx_mask, values=ctx)
        self._summ_tm1 = summaries

        out_vec = core_out

        ret = tuple()
        if self.return_outvecs:
            ret += (out_vec,)
        else:
            out_scores = self.out(out_vec)
            ret += (out_scores,)

        if self.return_alphas:
            ret += (alphas,)
        if self.return_scores:
            ret += (scores,)
        if self.return_other:
            ret += (embs, core_out, summaries)
        return ret[0] if len(ret) == 1 else ret









# region Encoders
class FastLSTMEncoderLayer(torch.nn.Module):
    """ Fast LSTM encoder layer using torch's built-in fast LSTM.
        Provides a more convenient interface.
        States are stored in .y_n and .c_n (initial states in .y_0 and .c_0)"""
    def __init__(self, indim, dim, bidir=False, dropout_in=0., bias=True, **kw):
        super(FastLSTMEncoderLayer, self).__init__(**kw)
        self.layer = torch.nn.LSTM(input_size=indim, hidden_size=dim, num_layers=1,
                                   bidirectional=bidir, bias=bias, batch_first=True)
        self.y_0 = q.val(torch.zeros((1 if not bidir else 2), 1, dim)).v
        self.c_0 = q.val(torch.zeros((1 if not bidir else 2), 1, dim)).v
        self.dropout_in = q.TimesharedDropout(dropout_in)
        self.y_n = None
        self.c_n = None

    def apply_dropouts(self, vecs):
        vecs = self.dropout_in(vecs)
        return vecs

    def forward(self, vecs, mask=None):
        """ if mask is not None, vecs are packed using PackedSequences
            and unpacked before outputting.
            Output sequence lengths can thus be shorter than input sequence lengths when providing mask """
        batsize = vecs.size(0)
        vecs = self.apply_dropouts(vecs)
        if mask is not None:
            vecs, order = q.seq_pack(vecs, mask=mask)
        y_0 = self.y_0.repeat(1, batsize, 1)
        c_0 = self.c_0.repeat(1, batsize, 1)
        out, (y_n, c_n) = self.layer(vecs, (y_0, c_0))
        if mask is not None:
            y_n = y_n.index_select(1, order)
            c_n = c_n.index_select(1, order)
            out, rmask = q.seq_unpack(out, order)
            # assert((mask - rmask).float().norm().cpu().data[0] == 0)
        self.y_n = y_n.transpose(1, 0)
        self.c_n = c_n.transpose(1, 0)      # batch-first
        return out


class FastGRUEncoderLayer(torch.nn.Module):
    """ Fast GRU encoder layer using torch's built-in fast GRU.
        Provides a more convenient interface.
        State is stored in .h_n (initial state in .h_0)"""
    def __init__(self, indim, dim, bidir=False, dropout_in=0., bias=True, **kw):
        super(FastGRUEncoderLayer, self).__init__(**kw)
        self.layer = torch.nn.GRU(input_size=indim, hidden_size=dim, num_layers=1,
                                   bidirectional=bidir, bias=bias, batch_first=True)
        self.h_0 = q.val(torch.zeros((1 if not bidir else 2), 1, dim)).v
        self.dropout_in = q.TimesharedDropout(dropout_in)
        self.h_n = None

    def forward(self, vecs, mask=None):
        batsize = vecs.size(0)
        vecs = self.dropout_in(vecs)
        if mask is not None:
            vecs, order = q.seq_pack(vecs, mask=mask)
        h_0 = self.h_0.repeat(1, batsize, 1)
        out, h_n = self.layer(vecs, h_0)
        if mask is not None:
            h_n = h_n.index_select(1, order)
            out, rmask = q.seq_unpack(out, order)
        self.h_n = h_n.transpose(1, 0)      # batch-first
        return out


class FastLSTMEncoder(torch.nn.Module):
    """ Fast LSTM encoder using multiple layers.
        !! every layer packs and unpacks a PackedSequence --> might be inefficient
        Access to states is provided through .y_n, .y_0, .c_n and .c_0 (bottom layer first) """
    def __init__(self, indim, *dims, **kw):
        self.bidir = q.getkw(kw, "bidir", default=False)
        self.dropout = q.getkw(kw, "dropout_in", default=0.)
        self.bias = q.getkw(kw, "bias", default=True)
        super(FastLSTMEncoder, self).__init__(**kw)
        if not q.issequence(dims):
            dims = (dims,)
        dims = (indim,) + dims
        self.dims = dims
        self.layers = torch.nn.ModuleList()
        self.make_layers()

    def make_layers(self):
        for i in range(1, len(self.dims)):
            layer = FastLSTMEncoderLayer(indim=self.dims[i-1] * (1 if not self.bidir or i == 1 else 2),
                                        dim=self.dims[i], bidir=self.bidir, dropout_in=self.dropout, bias=self.bias)
            self.layers.append(layer)

    # region state management
    @property
    def y_n(self):
        acc = [layer.y_n for layer in self.layers]      # bottom layers first
        return acc

    @property
    def c_n(self):
        return [layer.c_n for layer in self.layers]

    @property
    def y_0(self):
        return [layer.y_0 for layer in self.layers]

    @y_0.setter
    def y_0(self, *values):
        for layer, value in zip(self.layers, values):
            layer.y_0 = value

    @property
    def c_0(self):
        return [layer.c_0 for layer in self.layers]

    @c_0.setter
    def c_0(self, *values):
        for layer, value in zip(self.layers, values):
            layer.c_0 = value
    # endregion

    def forward(self, x, mask=None):
        out = x
        for layer in self.layers:
            out = layer(out, mask=mask)
        return out


class FastGRUEncoder(torch.nn.Module):
    """ Fast LSTM encoder using multiple layers.
        !! every layer packs and unpacks a PackedSequence --> might be inefficient
        Access to states of layer is provided through .h_0 and .h_n (bottom layers first) """
    def __init__(self, indim, *dims, **kw):
        self.bidir = q.getkw(kw, "bidir", default=False)
        self.dropout = q.getkw(kw, "dropout_in", default=0.)
        self.bias = q.getkw(kw, "bias", default=True)
        super(FastGRUEncoder, self).__init__(**kw)
        if not q.issequence(dims):
            dims = (dims,)
        dims = (indim,) + dims
        self.dims = dims
        self.layers = torch.nn.ModuleList()
        self.make_layers()

    def make_layers(self):
        for i in range(1, len(self.dims)):
            layer = FastGRUEncoderLayer(indim=self.dims[i-1] * (1 if not self.bidir or i == 1 else 2),
                                        dim=self.dims[i], bidir=self.bidir, dropout_in=self.dropout, bias=self.bias)
            self.layers.append(layer)

    # region state management
    @property
    def h_n(self):
        acc = [layer.h_n for layer in self.layers]      # bottom layers first
        return acc

    @property
    def h_0(self):
        return [layer.h_0 for layer in self.layers]

    @h_0.setter
    def h_0(self, *values):
        for layer, value in zip(self.layers, values):
            layer.h_0 = value
    # endregion

    def forward(self, x, mask=None):
        out = x
        for layer in self.layers:
            out = layer(out, mask=mask)
        return out


class FastestGRUEncoderLayer(torch.nn.Module):
    """ Fastest GRU encoder layer using torch's built-in fast GRU.
        Provides a more convenient interface.
        State is stored in .h_n (initial state in .h_0).
        !!! Dropout_in, dropout_rec are shared among all examples in a batch (and across timesteps) !!!"""
    def __init__(self, indim, dim, bidir=False, dropout_in=0., dropout_rec=0., bias=True, **kw):
        super(FastestGRUEncoderLayer, self).__init__(**kw)
        this = self

        class GRUOverride(torch.nn.GRU):
            @property
            def all_weights(self):
                acc = []
                for weights in self._all_weights:
                    iacc = []
                    for weight in weights:
                        if hasattr(this, weight) and getattr(this, weight) is not None:
                            iacc.append(getattr(this, weight))
                        else:
                            iacc.append(getattr(self, weight))
                    acc.append(iacc)
                return acc
                # return [[getattr(this, weight) for weight in weights] for weights in self._all_weights]

        self.layer = GRUOverride(input_size=indim, hidden_size=dim, num_layers=1,
                                   bidirectional=bidir, bias=bias, batch_first=True)
        self.h_0 = q.val(torch.zeros((1 if not bidir else 2), dim)).v
        self.dropout_in = torch.nn.Dropout(dropout_in) if dropout_in > 0 else None
        self.dropout_rec = torch.nn.Dropout(dropout_rec) if dropout_rec > 0 else None

        self.h_n = None

        # weight placeholders
        for weights in self.layer._all_weights:
            for weight in weights:
                setattr(self, weight, None)

    def forward(self, vecs, mask=None, order=None, batsize=None, h_0=None, ret_states=False):
        batsize = vecs.size(0) if batsize is None else batsize

        # dropouts
        if self.training and self.dropout_in is not None:
            weights = ["weight_ih_l0", "weight_ih_l0_reverse"]
            weights = filter(lambda x: hasattr(self, x), weights)
            for weight in weights:
                dropoutmask = torch.ones(getattr(self.layer, weight).size(1)).to(vecs.device)
                dropoutmask = self.dropout_in(dropoutmask)
                new_weight_ih = getattr(self.layer, weight) * dropoutmask.unsqueeze(0)
                setattr(self, weight, new_weight_ih)
        if self.training and self.dropout_rec is not None:
            weights = ["weight_hh_l0", "weight_hh_l0_reverse"]
            weights = filter(lambda x: hasattr(self, x), weights)
            for weight in weights:
                dropoutmask = torch.ones(getattr(self.layer, weight).size(1)).to(vecs.device)
                dropoutmask = self.dropout_rec(dropoutmask)
                new_weight_hh = getattr(self.layer, weight) * dropoutmask.unsqueeze(0)
                setattr(self, weight, new_weight_hh)

        if mask is not None:
            assert(not isinstance(vecs, torch.nn.utils.rnn.PackedSequence))
            vecs, order = q.seq_pack(vecs, mask=mask)

        # init states
        if h_0 is not None:
            if h_0.dim() == 3:
                h_0 = h_0.transpose(1, 0)   # h_0 kwargs are given batch-first
        else:
            h_0 = self.h_0
        if h_0.dim() == 2:
            h_0 = self.h_0.unsqueeze(1).repeat(1, batsize, 1)

        # apply
        out, h_n = self.layer(vecs, h_0)
        if order is not None:
            h_n = h_n.index_select(1, order)
        if mask is not None:
            out, rmask = q.seq_unpack(out, order)

        h_n = h_n.transpose(1, 0)      # batch-first
        self.h_n = h_n
        if ret_states:
            return out, h_n
        else:
            return out


class FastestGRUEncoder(FastGRUEncoder):        # TODO: TEST
    """ Fastest GRU encoder using multiple layers.
        Access to states of layer is provided through .h_0 and .h_n (bottom layers first) """
    def __init__(self, indim, *dims, **kw):
        self.dropout_rec = q.getkw(kw, "dropout_rec", default=0.)
        super(FastestGRUEncoder, self).__init__(indim, *dims, **kw)

    def make_layers(self):
        for i in range(1, len(self.dims)):
            layer = FastestGRUEncoderLayer(indim=self.dims[i-1] * (1 if not self.bidir or i == 1 else 2),
                                        dim=self.dims[i], bidir=self.bidir, dropout_in=self.dropout,
                                        dropout_rec=self.dropout_rec, bias=self.bias)
            self.layers.append(layer)

    def forward(self, x, mask=None, batsize=None, h_0s=None, ret_states=False):
        batsize = x.size(0) if batsize is None else batsize
        imask = mask
        order = None
        if mask is not None:
            assert (not isinstance(x, torch.nn.utils.rnn.PackedSequence))
            x, order = q.seq_pack(x, mask=mask)
            imask = None
        out = x

        h_0s = [] if h_0s is None else list(h_0s)
        assert(len(h_0s) <= len(self.layers))
        h_0s = [None] * (len(self.layers) - len(h_0s)) + h_0s

        states_to_ret = []

        for layer, h_0 in zip(self.layers, h_0s):
            out = layer(out, mask=imask, batsize=batsize, h_0=h_0)
            h_n = layer.h_n
            if order is not None:
                h_n = h_n.index_select(0, order)
                layer.h_n = h_n
            states_to_ret.append((h_n,))

        if mask is not None:
            out, rmask = q.seq_unpack(out, order)

        if ret_states:
            return out, states_to_ret
        else:
            return out


class FastestLSTMEncoderLayer(torch.nn.Module):
    """ Fastest LSTM encoder layer using torch's built-in fast LSTM.
        Provides a more convenient interface.
        States are stored in .y_n and .c_n (initial states in .y_0 and .c_0).
        !!! Dropout_in, dropout_rec are shared among all examples in a batch (and across timesteps) !!!"""
    def __init__(self, indim, dim, bidir=False, dropout_in=0., dropout_rec=0., bias=True, skipper=False, **kw):
        super(FastestLSTMEncoderLayer, self).__init__(**kw)
        self.skipper = skipper      # TODO
        this = self

        class LSTMOverride(torch.nn.LSTM):
            @property
            def all_weights(self):
                acc = []
                for weights in self._all_weights:
                    iacc = []
                    for weight in weights:
                        if hasattr(this, weight) and getattr(this, weight) is not None:
                            iacc.append(getattr(this, weight))
                        else:
                            iacc.append(getattr(self, weight))
                    acc.append(iacc)
                return acc
                # return [[getattr(this, weight) for weight in weights] for weights in self._all_weights]

        self.layer = LSTMOverride(input_size=indim, hidden_size=dim, num_layers=1,
                                   bidirectional=bidir, bias=bias, batch_first=True)
        self.y_0 = q.val(torch.zeros((1 if not bidir else 2), dim)).v
        self.c_0 = q.val(torch.zeros((1 if not bidir else 2), dim)).v
        self.dropout_in = torch.nn.Dropout(dropout_in) if dropout_in > 0 else None
        self.dropout_rec = torch.nn.Dropout(dropout_rec) if dropout_rec > 0 else None

        self.y_n = None
        self.c_n = None

        # weight placeholders
        for weights in self.layer._all_weights:
            for weight in weights:
                setattr(self, weight, None)

    def forward(self, vecs, mask=None, batsize=None, y_0=None, c_0=None, ret_states=False):
        batsize = vecs.size(0) if batsize is None else batsize
        order = None

        # dropouts
        if self.dropout_in is not None:
            weights = ["weight_ih_l0", "weight_ih_l0_reverse"]
            weights = filter(lambda x: hasattr(self, x), weights)
            for weight in weights:
                layer_weight = getattr(self.layer, weight)
                dropoutmask = torch.ones(layer_weight.size(1)).to(layer_weight.device)
                dropoutmask = self.dropout_in(dropoutmask)
                new_weight_ih = getattr(self.layer, weight) * dropoutmask.unsqueeze(0)
                setattr(self, weight, new_weight_ih)
        if self.dropout_rec is not None:
            weights = ["weight_hh_l0", "weight_hh_l0_reverse"]
            weights = filter(lambda x: hasattr(self, x), weights)
            for weight in weights:
                layer_weight = getattr(self.layer, weight)
                dropoutmask = torch.ones(layer_weight.size(1)).to(layer_weight.device)
                dropoutmask = self.dropout_rec(dropoutmask)
                new_weight_hh = getattr(self.layer, weight) * dropoutmask.unsqueeze(0)
                setattr(self, weight, new_weight_hh)

        if mask is not None:
            assert(not isinstance(vecs, torch.nn.utils.rnn.PackedSequence))
            vecs, order = q.seq_pack(vecs, mask=mask)

        # init states
        if y_0 is not None:
            if y_0.dim() == 3:
                y_0 = y_0.transpose(1, 0)   # y_0 kwargs are given batch-first
        else:
            y_0 = self.y_0
        if c_0 is not None:
            if c_0.dim() == 3:
                c_0 = c_0.transpose(1, 0)
        else:
            c_0 = self.c_0
        if y_0.dim() == 2:
            y_0 = y_0.unsqueeze(1).repeat(1, batsize, 1)
        if c_0.dim() == 2:
            c_0 = c_0.unsqueeze(1).repeat(1, batsize, 1)

        # apply
        out, (y_n, c_n) = self.layer(vecs, (y_0, c_0))

        # use order
        if order is not None:
            y_n = y_n.index_select(1, order)
            c_n = c_n.index_select(1, order)
        if mask is not None:
            out, rmask = q.seq_unpack(out, order)

        y_n = y_n.transpose(1, 0)       # output states must be batch first
        c_n = c_n.transpose(1, 0)
        self.y_n = y_n
        self.c_n = c_n
        if ret_states:
            return out, (y_n, c_n)
        else:
            return out


class FastestLSTMEncoder(FastLSTMEncoder):
    """ Fastest LSTM encoder using multiple layers.
        Access to states of layers is provided through .y_0, .c_0 and .y_n, .c_n (bottom layers first) """
    def __init__(self, indim, *dims, **kw):
        self.dropout_rec = q.getkw(kw, "dropout_rec", default=0.)
        self.skipper = q.getkw(kw, "skipper", default=False)
        super(FastestLSTMEncoder, self).__init__(indim, *dims, **kw)

    def make_layers(self):
        for i in range(1, len(self.dims)):
            layer = FastestLSTMEncoderLayer(indim=self.dims[i-1] * (1 if not self.bidir or i == 1 else 2),
                                        dim=self.dims[i], bidir=self.bidir, dropout_in=self.dropout,
                                        dropout_rec=self.dropout_rec, bias=self.bias, skipper=self.skipper)
            self.layers.append(layer)

    def forward(self, x, mask=None, batsize=None, y_0s=None, c_0s=None, ret_states=False):
        """ top layer states return last """
        batsize = x.size(0) if batsize is None else batsize
        imask = mask
        order = None
        if mask is not None:
            assert (not isinstance(x, torch.nn.utils.rnn.PackedSequence))
            x, order = q.seq_pack(x, mask=mask)
            imask = None
        out = x

        # init states
        y_0s = [] if y_0s is None else list(y_0s)
        assert(len(y_0s) <= len(self.layers))
        y_0s = [None] * (len(self.layers) - len(y_0s)) + y_0s
        c_0s = [] if c_0s is None else list(c_0s)
        assert(len(c_0s) <= len(self.layers))
        c_0s = [None] * (len(self.layers) - len(c_0s)) + c_0s

        states_to_ret = []

        for layer, y0, c0 in zip(self.layers, y_0s, c_0s):
            out = layer(out, mask=imask, batsize=batsize, y_0=y0, c_0=c0)
            y_i_n, c_i_n = layer.y_n, layer.c_n
            if order is not None:
                y_i_n = y_i_n.index_select(0, order)
                c_i_n = c_i_n.index_select(0, order)
                layer.y_n = y_i_n
                layer.c_n = c_i_n
            states_to_ret.append((y_i_n, c_i_n))

        if mask is not None:
            out, rmask = q.seq_unpack(out, order)

        if ret_states:
            return out, states_to_ret
        else:
            return out

# endregion


class AutoMaskedOut(torch.nn.Module):
    def __init__(self, baseout, automasker, **kw):
        super(AutoMaskedOut, self).__init__(**kw)
        self.automasker = automasker
        self.baseout = baseout

    def forward(self, *args, **kw):
        assert ("mask" not in kw)
        mask = self.automasker.get_out_mask()
        kw["mask"] = mask
        ret = self.baseout(*args, **kw)
        return ret


class AutoMasker(object):
    def __init__(self, rules, inpD, outD, **kw):
        super(AutoMasker, self).__init__(**kw)
        self.rules = rules
        self.inpD, self.outD = inpD, outD

    def update(self, x):
        """ updates automasker with next element in the sequence """
        pass    # TODO

    def get_out_mask(self):
        """ returns a mask over outD """
        pass    # TODO