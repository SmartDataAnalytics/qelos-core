import qelos_core as q
import torch
import numpy as np


class PointerGeneratorOut(torch.nn.Module):
    def __init__(self, gen_prob_comp, base_out, **kw):
        """
        :param gen_prob_comp:   module used to generate probability of generating vs pointing
        :param base_out:        base out layer generating scores (not probs!) for generator tokens
        :param kw:
        """
        super(PointerGeneratorOut, self).__init__(**kw)
        self.base_out = base_out
        self.gen_prob_comp = gen_prob_comp
        self.sm = torch.nn.Softmax(-1)

    def forward(self, x, alphas, ctx_map):
        """

        :param x:
        :param alphas:      (batsize, seqlen)
        :param ctx_map:     (batsize, seqlen) integer ids in outvoc space
        :return:
        """
        interp = self.gen_prob_comp(x)

        gen_probs = self.base_out(x)        # (batsize, outvocsize)
        gen_probs = self.sm(gen_probs)

        ptr_probs = torch.zeros_like(gen_probs)
        ptr_probs.scatter_(1, ctx_map, alphas)      # alphas: already normalized

        out_probs = interp * gen_probs + (1 - interp) * ptr_probs
        return out_probs


class PointerGeneratorCell(q.DecoderCell):
    def __init__(self, emb, core, att, out, feed_att=False, summ_0=None, **kw):
        assert(isinstance(out, PointerGeneratorOut))
        super(PointerGeneratorCell, self).__init__(emb, core, att, out, feed_att=feed_att, summ_0=summ_0, **kw)

    def forward(self, x_t, ctx=None, ctx_mask=None, ctx_map=None, **kw):
        """
        :param x_t:
        :param ctx:
        :param ctx_mask:
        :param ctx_map:     maps ctx positions to words in inp dic
        :param kw:
        :return:
        """
        assert(ctx is not None)
        assert(ctx_map is not None)
        embs = self.emb(x_t)
        if q.issequence(embs) and len(embs) == 2:
            embs, mask = embs

        core_inp = embs
        if self.feed_att:
            core_inp = torch.cat([core_inp, self.summ_tm1], 1)
        core_out = self.core(core_inp)

        alphas, summaries = self.att(core_out, ctx, ctx_mask=ctx_mask, values=ctx)
        self.summ_tm1 = summaries

        to_out = []
        if self.use_cell_out:
            to_out.append(acts)
        if self.use_att_sum:
            to_out.append(summaries)
        if self.use_x_t_emb:
            to_out.append(embs)
        to_out = torch.cat(to_out, 1)

        outscores = self.out(to_out, alphas, ctx_map)
        return outscores


def run(lr=0.001):
    pass

