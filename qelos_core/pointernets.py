import qelos_core as q
import torch
import numpy as np


class PointerGeneratorOut(torch.nn.Module):
    """ Uses sum for overlaps ! (scatter_add_)"""
    def __init__(self, outdic, gen_prob_comp, gen_out, inpdic=None, gen_zero=None, gen_outD=None, **kw):
        """
        :param outdic:          output dictionary, must contain all tokens in inpdic and gen_out.D
        :param gen_prob_comp:   module to compute probability of generating vs pointing
        :param gen_out:         module to compute generation probabilities.
                                    must have a dictionary accessible as ".D".
                                    must produce normalized probabilities (use softmax)
        :param inpdic:          input dictionary (for pointer)
        :param gen_zero:        None or set of tokens for which the gen_out's prob will be set to zero.
                                All tokens should occur in inpdic (or their score will always be zero)
        :param gen_outD:        if set, gen_out must not have a ".D"
        :param kw:
        """
        super(PointerGeneratorOut, self).__init__(**kw)
        self.gen_out = gen_out
        self.gen_outD = self.gen_out.D if gen_outD is None else gen_outD
        self.gen_prob_comp = gen_prob_comp
        self.outsize = max(outdic.values()) + 1
        self.gen_to_out = q.val(torch.zeros(1, max(self.gen_outD.values()) + 1, dtype=torch.int64)).v
        self.gen_zero_mask = None if gen_zero is None else \
            q.val(torch.ones_like(self.gen_to_out, dtype=torch.float32)).v
        # (1, genvocsize), integer ids in outvoc, one-to-one mapping
        # if symbol in gendic is not in outdic, throws error
        for k, v in self.gen_outD.items():
            if k in outdic:
                self.gen_to_out[0, v] = outdic[k]
                if gen_zero is not None:
                    if k in gen_zero:
                        self.gen_zero_mask[0, v] = 0
            else:
                raise q.SumTingWongException("symbols in gen_outD must be in outdic, but \"{}\" isn't".format(k))

        self.inp_to_out = q.val(torch.zeros(max(inpdic.values()) + 1, dtype=torch.int64)).v
        # (1, inpvocsize), integer ids in outvoc, one-to-one mapping
        # if symbol in inpdic is not in outdic, throws error
        for k, v in inpdic.items():
            if k in outdic:
                self.inp_to_out[v] = outdic[k]
            else:
                raise q.SumTingWongException("symbols in inpdic must be in outdic, but \"{}\" isn't".format(k))

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
        # TODO: why is <MASK> prob always zero?
        if self.gen_zero_mask is not None:
            gen_probs = gen_probs * self.gen_zero_mask
        out_probs_gen = torch.zeros(x.size(0), self.outsize, dtype=x.dtype, device=x.device)
        out_probs_gen.scatter_add_(1, self.gen_to_out.repeat(gen_probs.size(0), 1), gen_probs)

        ctx_out = self.inp_to_out[ctx_inp]      # map int ids in inp voc to out voc
        out_probs_ptr = torch.zeros(x.size(0), self.outsize, dtype=x.dtype, device=x.device)
        out_probs_ptr.scatter_add_(1, ctx_out[:, :alphas.size(1)], alphas)

        out_probs = interp * out_probs_gen + (1 - interp) * out_probs_ptr

        return out_probs


class PointerGeneratorCell(q.LuongCell):
    def __init__(self, emb=None, core=None, att=None, merge=None, out=None, feed_att=False, **kw):
        super(PointerGeneratorCell, self).__init__(emb=emb, core=core, att=att, merge=merge, out=None,
                                                   feed_att=feed_att, return_alphas=True, **kw)
        # assert(isinstance(out, PointerGeneratorOut))
        self.pointer_out = out

    def forward(self, x_t, ctx=None, ctx_mask=None, ctx_inp=None, **kw):
        """
        :param x_t:         (batsize,) integer ids in outvoc
        :param ctx:         (batsize, seqlen, dim) ctx to use for attention
        :param ctx_mask:    (batsize, seqlen) bool mask for ctx
        :param ctx_inp:     (batsize, seqlen) integer ids in inpvoc = original input used to compute ctx
        :param kw:
        :return:
        """
        assert(ctx_inp is not None)
        out_vec, alphas = super(PointerGeneratorCell, self).forward(x_t, ctx=ctx, ctx_mask=ctx_mask, **kw)
        outscores = self.pointer_out(out_vec, alphas, ctx_inp)
        return outscores


def run(lr=0.001):
    pass

