import qelos_core as q
import torch
import numpy as np


class PointerGeneratorOut(torch.nn.Module):
    """ Uses sum for overlaps ! (scatter_add_)"""
    def __init__(self, outdic, gen_prob_comp, gen_out, inpdic=None, out_logits=True, **kw):
        """
        :param outdic:          output dictionary, must contain all symbols in inpdic and gen_out.D
        :param gen_prob_comp:   module to compute probability of generating vs pointing
        :param gen_out:         module to compute generation probabilities.
                                    must have a dictionary accessible as ".D".
                                    must produce normalized probabilities (use softmax)
        :param inpdic:          input dictionary (for pointer)
        :param out_logits:      apply log on final probs (if True, can use NLLLoss)
        :param kw:
        """
        super(PointerGeneratorOut, self).__init__(**kw)
        self.gen_out = gen_out
        self.gen_prob_comp = gen_prob_comp
        self.outsize = max(outdic.values())
        self.gen_to_out = q.val(torch.zeros(1, max(gen_out.D.values()), dtype=torch.int64)).v
        # (1, genvocsize), integer ids in outvoc, one-to-one mapping
        # if symbol in gendic is not in outdic, throws error
        for k, v in gen_out.D.items():
            if k not in outdic:
                raise q.SumTingWongException("symbols in gen_out.D must be in outdic, but \"{}\" isn't".format(k))
            self.gen_to_out[0, v] = outdic[k]
        self.inp_to_out = q.val(torch.zeros(max(inpdic.values()), dtype=torch.int64)).v
        # (1, inpvocsize), integer ids in outvoc, one-to-one mapping
        # if symbol in inpdic is not in outdic, throws error
        for k, v in inpdic.items():
            if k not in outdic:
                raise q.SumTingWongException("symbols in inpdic must be in outdic, but \"{}\" isn't".format(k))
            self.inp_to_out[v] = outdic[k]

        self.out_logits = out_logits

    def forward(self, x, alphas, ctx_inp):
        """
        :param x:       input for this time step
                            (batsize, outdim) floats
        :param alphas:  normalized probabilities over ctx
                            (batsize, seqlen) floats
        :param ctx_inp: input used to compute context and alphas
                            (batsize, seqlen) integer ids in inpvoc
        :return:
        """
        interp = self.gen_prob_comp(x)

        gen_probs = self.gen_out(x)        # (batsize, outvocsize)
        out_probs_gen = torch.zeros(x.size(0), self.outsize, dtype=x.dtype, device=x.device)
        out_probs_gen.scatter_add_(1, self.gen_to_out.repeat(gen_probs.size(0), 1), gen_probs)

        ctx_out = self.inp_to_out[ctx_inp]      # map int ids in inp voc to out voc
        out_probs_ptr = torch.zeros(x.size(0), self.outsize, dtype=x.dtype, device=x.device)
        out_probs_ptr.scatter_add_(1, ctx_out, alphas)

        out_probs = interp * out_probs_gen + (1 - interp) * out_probs_ptr

        if self.out_logits:
            out_probs = torch.log(out_probs)
        return out_probs


class PointerGeneratorCell(q.DecoderCell):
    def __init__(self, emb, core, att, out, feed_att=False, summ_0=None, **kw):
        """
        :param emb:
        :param core:
        :param att:
        :param out:
        :param feed_att:
        :param summ_0:
        :param kw:
        """
        assert(isinstance(out, PointerGeneratorOut))
        super(PointerGeneratorCell, self).__init__(emb, core, att, out, feed_att=feed_att, summ_0=summ_0, **kw)

    def forward(self, x_t, ctx=None, ctx_mask=None, ctx_inp=None, **kw):
        """
        :param x_t:         (batsize,) integer ids in outvoc
        :param ctx:         (batsize, seqlen, dim) ctx to use for attention
        :param ctx_mask:    (batsize, seqlen) bool mask for ctx
        :param ctx_inp:     (batsize, seqlen) integer ids in inpvoc = original input used to compute ctx
        :param kw:
        :return:
        """
        assert(ctx is not None)
        assert(ctx_inp is not None)
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
            to_out.append(core_out)
        if self.use_att_sum:
            to_out.append(summaries)
        if self.use_x_t_emb:
            to_out.append(embs)
        to_out = torch.cat(to_out, 1)

        outscores = self.out(to_out, alphas, ctx_inp)
        return outscores


def run(lr=0.001):
    pass

