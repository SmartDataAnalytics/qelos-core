import qelos_core as q
import torch
import numpy as np


class PointerGeneratorOut(torch.nn.Module):
    def __init__(self, outdic, gen_out, inpdic=None, gen_zero=None, gen_outD=None, **kw):
        """
                :param outdic:          output dictionary, must contain all tokens in inpdic and gen_out.D
                :param gen_prob_comp:   module to compute probability of generating vs pointing
                                        must produce (batsize, 1) shapes
                :param gen_out:         module to compute generation scores.
                                            must have a dictionary accessible as ".D".
                                            must produce unnormalized scores (no softmax)
                :param inpdic:          input dictionary (for pointer)
                :param gen_zero:        None or set of tokens for which the gen_out's prob will be set to zero.
                                        All tokens should occur in inpdic (or their score will always be zero)
                :param gen_outD:        if set, gen_out must not have a ".D"
                :param kw:
                """
        super(PointerGeneratorOut, self).__init__(**kw)
        self.gen_out = gen_out
        self.D = outdic
        self.gen_outD = self.gen_out.D if gen_outD is None else gen_outD
        self.outsize = max(outdic.values()) + 1
        self.gen_to_out = q.val(torch.zeros(1, max(self.gen_outD.values()) + 1, dtype=torch.int64)).v
        # --> where in out to scatter every element of the gen
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
        # --> where in out to scatter every element of the inp
        # (1, inpvocsize), integer ids in outvoc, one-to-one mapping
        # if symbol in inpdic is not in outdic, throws error
        for k, v in inpdic.items():
            if k in outdic:
                self.inp_to_out[v] = outdic[k]
            else:
                raise q.SumTingWongException("symbols in inpdic must be in outdic, but \"{}\" isn't".format(k))
        self.sm = torch.nn.Softmax(-1)
        self._reset()
        self.check()

    def check(self):
        pass

    def _reset(self):
        pass


class PointerGeneratorOutSeparate(PointerGeneratorOut):
    """ Uses sum for overlaps ! (scatter_add_)"""
    def __init__(self, outdic, gen_prob_comp, gen_out,
                 inpdic=None, gen_zero=None, gen_outD=None, **kw):
        """
        :param outdic:          output dictionary, must contain all tokens in inpdic and gen_out.D
        :param gen_prob_comp:   module to compute probability of generating vs pointing
                                must produce (batsize, 1) shapes
        :param gen_out:         module to compute generation scores.
                                    must have a dictionary accessible as ".D".
                                    must produce unnormalized scores (no softmax)
        :param inpdic:          input dictionary (for pointer)
        :param gen_zero:        None or set of tokens for which the gen_out's prob will be set to zero.
                                All tokens should occur in inpdic (or their score will always be zero)
        :param gen_outD:        if set, gen_out must not have a ".D"
        :param kw:
        """
        super(PointerGeneratorOutSeparate, self).__init__(outdic, gen_out,
                  inpdic=inpdic, gen_zero=gen_zero, gen_outD=gen_outD, **kw)
        self.gen_prob_comp = gen_prob_comp

    def forward(self, x, scores, ctx_inp, mask=None, **kw):
        """
        :param x:       input for this time step
                            (batsize, outdim) floats
        :param scores:  unnormalized scores over ctx
                            (batsize, seqlen) floats
        :param ctx_inp: input used to compute context and alphas
                            (batsize, seqlen) integer ids in inpvoc
        :param mask:    mask on the output tokens
                            (batsize, outvocsize) one or zero
        :return:
        """
        mask = mask.float() if mask is not None else None
        interp = self.gen_prob_comp(x)
        assert((interp.clamp(min=0, max=1) == interp).all().cpu().item() == 1)
        batsize = x.size(0)

        gen_scores = self.gen_out(x)        # (batsize, outvocsize)

        #region masks
        if self.gen_zero_mask is not None:
            gen_scores = gen_scores + torch.log(self.gen_zero_mask)
        if mask is not None:
            gen_mask = torch.gather(mask, 1, self.gen_to_out.repeat(batsize, 1))
            gen_scores = gen_scores + torch.log(gen_mask.float())

        gen_infty_mask = (gen_scores != -float("inf")).float().sum(1).unsqueeze(1) == 0
        if gen_infty_mask.any().cpu().item() == 1:  # if any gen scores became all-neg-infinite
            interp = interp * (1 - gen_infty_mask.float()) + gen_infty_mask.float() * torch.zeros_like(interp)
            gen_scores = torch.gather(torch.cat([gen_scores.unsqueeze(2), torch.ones_like(gen_scores).unsqueeze(2)], 2),
                                  2, gen_infty_mask.long().unsqueeze(2).repeat(1, gen_scores.size(1), 1))\
                                    .squeeze(2)
        # endregion

        gen_probs = self.sm(gen_scores)
        out_probs_gen = torch.zeros(x.size(0), self.outsize, dtype=x.dtype, device=x.device)
        out_probs_gen.scatter_add_(1, self.gen_to_out.repeat(batsize, 1), gen_probs)

        # region masks
        if mask is not None:
            # mask is on outdic --> transform to inpdic --> transform to inp seq (!=inpdic)
            inp_mask = torch.gather(mask, 1, self.inp_to_out.repeat(batsize, 1))
            inp_mask = torch.gather(inp_mask, 1, ctx_inp[:, :scores.size(1)])
            scores = scores + torch.log(inp_mask.float())

        inp_infty_mask = (scores != -float("inf")).float().sum(1).unsqueeze(1) == 0
        if inp_infty_mask.any().cpu().item() == 1:  # if any gen scores became all-neg-infinite
            assert(((gen_infty_mask.float() + inp_infty_mask.float()) < 2).all().cpu().item() == 1)
            interp = interp * (1 - inp_infty_mask.float()) + inp_infty_mask.float() * torch.ones_like(interp)
            scores = torch.gather(
                torch.cat([scores.unsqueeze(2), torch.ones_like(scores).unsqueeze(2)], 2),
                2, inp_infty_mask.long().unsqueeze(2).repeat(1, scores.size(1), 1)) \
                .squeeze(2)
        # endregion

        alphas = self.sm(scores)
        ctx_out = self.inp_to_out[ctx_inp]      # map int ids in inp voc to out voc
        out_probs_ptr = torch.zeros(x.size(0), self.outsize, dtype=x.dtype, device=x.device)
        out_probs_ptr.scatter_add_(1, ctx_out[:, :alphas.size(1)], alphas)

        out_probs = interp * out_probs_gen + (1 - interp) * out_probs_ptr
        assert(out_probs.size() == (batsize, len(self.D)))

        return out_probs


class PointerGeneratorOutShared(PointerGeneratorOut):
    """ Uses sum for overlaps ! (scatter_add_)"""

    def forward(self, x, scores, ctx_inp, mask=None, **kw):
        """
        :param x:       input for this time step
                            (batsize, outdim) floats
        :param scores:  unnormalized scores over ctx
                            (batsize, seqlen) floats
        :param ctx_inp: input used to compute context and alphas
                            (batsize, seqlen) integer ids in inpvoc
        :param mask:    mask on the output tokens
                            (batsize, outvocsize) one or zero
        :return:
        """
        mask = mask.float() if mask is not None else None
        batsize = x.size(0)

        gen_scores = self.gen_out(x)        # (batsize, outvocsize)

        if self.gen_zero_mask is not None:
            gen_scores = gen_scores + torch.log(self.gen_zero_mask)

        out_scores_gen = torch.zeros(x.size(0), self.outsize, dtype=x.dtype, device=x.device)
        out_scores_gen_mask = torch.zeros_like(out_scores_gen)
        out_scores_gen.scatter_add_(1, self.gen_to_out.repeat(batsize, 1), gen_scores)
        out_scores_gen_mask.scatter_add_(1, self.gen_to_out.repeat(batsize, 1), torch.ones_like(gen_scores))

        ctx_out = self.inp_to_out[ctx_inp]      # map int ids in inp voc to out voc
        out_scores_ptr = torch.zeros(x.size(0), self.outsize, dtype=x.dtype, device=x.device)
        out_scores_ptr_mask = torch.zeros_like(out_scores_ptr)
        out_scores_ptr.scatter_add_(1, ctx_out[:, :scores.size(1)], scores)
        out_scores_ptr.scatter_add_(1, ctx_out[:, :scores.size(1)], torch.ones_like(scores))

        out_probs = out_scores_gen + out_scores_ptr
        out_probs_mask = (out_scores_ptr_mask + out_scores_gen_mask).clamp(min=0, max=1)
        out_probs = out_probs + torch.log(out_probs_mask)       # mask for not covered symbols in outD
        if mask is not None:
            out_probs = out_probs + torch.log(mask)
        out_probs = self.sm(out_probs)
        assert(out_probs.size() == (batsize, len(self.D)))

        return out_probs


class PointerGeneratorOutSharedMax(PointerGeneratorOut):
    def _reset(self):
        self.cached_inp_to_out = None

    def rec_reset(self):
        self._reset()

    def get_sparse_trans(self, trans, numout):
        """
        :param trans:   a scatter tensor mapping from indexes to some output indexes
                        (batsize, numinp)
        :return:
        """
        sparse_mapper = torch.zeros(trans.size(0), trans.size(1), numout)
        sparse_mapper.scatter_(2, trans.unsqueeze(2), 1)
        sparse_mapper[:, :, 0] = 0      # <MASK> zeroed
        #  TODO: might be not general enough
        return sparse_mapper

    def forward(self, x, scores, ctx_inp, mask=None, **kw):
        """
        :param x:       input for this time step
                            (batsize, outdim) floats
        :param scores:  unnormalized scores over ctx
                            (batsize, seqlen) floats
        :param ctx_inp: input used to compute context and alphas
                            (batsize, seqlen) integer ids in inpvoc
        :param mask:    mask on the output tokens
                            (batsize, outvocsize) one or zero
        :return:
        """
        mask = mask.float() if mask is not None else None
        batsize = x.size(0)

        gen_scores = self.gen_out(x)        # (batsize, outvocsize)

        if self.gen_zero_mask is not None:
            gen_scores = gen_scores + torch.log(self.gen_zero_mask)

        out_scores_gen = torch.zeros(x.size(0), self.outsize, dtype=x.dtype, device=x.device)
        out_scores_gen_mask = torch.zeros_like(out_scores_gen)
        out_scores_gen.scatter_add_(1, self.gen_to_out.repeat(batsize, 1), gen_scores)
        out_scores_gen_mask.scatter_(1, self.gen_to_out.repeat(batsize, 1), torch.ones_like(gen_scores))
        out_scores_gen = out_scores_gen + torch.log(out_scores_gen_mask)

        ctx_out = self.inp_to_out[ctx_inp]      # map int ids in inp voc to out voc
        # scatter max
        mapped_scores = torch.zeros(scores.size(0), scores.size(1), self.outsize, dtype=x.dtype, device=x.device)
        mapped_scores_mask = torch.zeros_like(mapped_scores)
        mapped_scores.scatter_(2, ctx_out[:, :scores.size(1)].unsqueeze(2), scores.unsqueeze(2))
        mapped_scores_mask.scatter_(2, ctx_out[:, :scores.size(1)].unsqueeze(2), 1)
        mapped_scores = mapped_scores + torch.log(mapped_scores_mask)
        out_scores_ptr, _ = mapped_scores.max(1)

        out_probs = torch.max(out_scores_gen, out_scores_ptr)
        if mask is not None:
            out_probs = out_probs + torch.log(mask)
        out_probs = self.sm(out_probs)
        assert(out_probs.size() == (batsize, len(self.D)))

        return out_probs


class PointerGeneratorCell(q.LuongCell):
    def __init__(self, emb=None, core=None, att=None, merge=None, out=None, feed_att=False, **kw):
        super(PointerGeneratorCell, self).__init__(emb=emb, core=core, att=att, merge=merge, out=None,
                                                   feed_att=feed_att, return_scores=True, **kw)
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

        if isinstance(self.pointer_out, q.AutoMaskedOut):
            self.pointer_out.update(x_t)

        out_vec, scores = super(PointerGeneratorCell, self).forward(x_t, ctx=ctx, ctx_mask=ctx_mask, **kw)
        outscores = self.pointer_out(out_vec, scores, ctx_inp)
        return outscores


def run(lr=0.001):
    pass

