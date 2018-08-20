import torch
import qelos_core as q
import numpy as np


class SketchyDecoderCell(q.LuongCell):
    def __init__(self, emb=None, core=None, att=None, merge=None, out=None, **kw):
        """
        :param emb:     embedder
        :param core:    core network for encoding partial decoding, should be a bidir rnn
        :param att:
        :param merge:
        :param out:
        :param kw:
        """
        super(SketchyDecoderCell, self).__init__(emb=emb, core=core, att=att, merge=merge, out=out, **kw)
        # self.encoder = q.FastestLSTMEncoder(*self.dims, bidir=True)

    def forward(self, x, whereat, ctx=None, ctx_mask=None):
        """
        :param x:       (batsize, out_seqlen) integer ids of the words in the partial tree so far
        :param whereat: (batsize,) integers, after which position index in x to insert predicted token
        :param ctx:     (batsize, in_seqlen, dim) the context to be used with attention
        :return:
        """
        if isinstance(self.out, q.AutoMaskedOut):
            self.out.update(x, whereat)

        mask = None
        embs = self.emb(x)
        if q.issequence(embs) and len(embs) == 2:
            embs, mask = embs

        if mask is None:
            mask = torch.ones_like(x[:, :, 0])
        y = self.encoder(x, mask=mask)
        whereat_sel = whereat.view(whereat.size(0), 1, 1).repeat(1, 1, y.size(2) // 2)
        z_fwd = y[:, :, :y.size(2)//2].gather(1, whereat_sel).squeeze(1)  # (batsize, dim)  # not completely correct but fine
        z_rev = y[:, :, y.size(2)//2:].gather(1, whereat_sel + 1).squeeze(1)
        z = torch.cat([z_fwd, z_rev], 1)

        core_inp = embs

        if self.att is not None:
            assert(ctx is not None)
            if self.feed_att:
                if self._h_hat_tm1 is None:
                    assert (self.h_hat_0 is not None)   #"h_hat_0 must be set when feed_att=True"
                    self._h_hat_tm1 = self.h_hat_0
                core_inp = torch.cat([core_inp, self._h_hat_tm1], 1)

        core_out = self.core(core_inp)

        alphas, summaries, scores = None, None, None
        out_vec = core_out
        if self.att is not None:
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


class SketchyEmbedder(torch.nn.Module):
    def __init__(self, content_emb, struct_emb, **kw):
        """
        :param content_emb:     embedder to use for embedding the content (token)
        :param struct_emb:      embedder to use for embedding the structure role of token (*/LS/NC/LSNC)
        :param kw:
        """
        super(SketchyEmbedder, self).__init__(**kw)
        self.content_emb = content_emb
        self.struct_emb = struct_emb

    def forward(self, x, x_role):
        emb, mask = self.content_emb(x)
        role_emb, _ = self.struct_emb(x_role)
        ret = torch.cat([emb, role_emb], emb.dim()-1)
        return ret, mask


class BFLRSketchyDecoderTF(q.TFDecoder):
    """ Breadth-first Left-Right Sketchy Decoder.
        Decodes higher levels first, decodes children left to right."""
    def forward(self, xs, whereats, **kw):
        """
        :param xs:      (batsize, seqlen, cond_seqlen) integer ids of tokens in tree (linearized).
        :param kw:
        :return:
        """
        # sequences along size=1 of xs can NOT be all-zero
        return super(BFLRSketchyDecoderTF, self).forward([xs, whereats])



def get_bflr_sketchy_tf_seqs(xs, levels, D):
    """ All inputs are numpy !
    :param xs:      (batsize, seqlen) integer ids of tokens in tree (linearized).
                    sequences must contain "<START> ... <END>" at level 0, which is considered as given
    :param levels:  (batsize, seqlen) integer ids of levels (0, 1, 2) of depth in tree
    """
    rD = {v: k for k, v in D.items()}
    seqses = []
    whereatses = []
    golds = []
    dims = [0, 0, 0]
    for k in range(len(xs)):
        seqs = []
        whereats = []
        gold = []
        for t in range(1, len(xs[k])):
            seq = []
            if rD[xs[k][t]] in ("<MASK>", "<END>"):
                pass
            else:
                gold.append(xs[k][t])
                # tokens from levels up and previous siblings must be in seq
                t_level = levels[k][t]
                c = 0
                for j in range(0, len(xs[k])):
                    if levels[k][j] < t_level or (j < t and levels[k][j] == t_level):
                        if rD[xs[k][j]] != "<MASK>":
                            seq.append(xs[k][j])
                            c += 1
                    if j == t:
                        whereats.append(c-1)
                seqs.append(seq)
            dims[2] = max(dims[2], len(seq))
        seqses.append(seqs)
        whereatses.append(whereats)
        golds.append(gold)
        dims[1] = max(dims[1], len(seqs))
    dims[0] = len(seqses)

    # matrices for output
    mat = np.zeros(tuple(dims), dtype="int64")
    # zero-length sequences must encode something
    mat[:, :, 0] = D["<START>"]
    mat[:, :, 1] = D["<END>"]
    for i, seqs in enumerate(seqses):
        for j, seq in enumerate(seqs):
            mat[i, j, :len(seq)] = seq
    whereatmat = np.zeros((dims[0], dims[1]), dtype="int64")
    for i, seq in enumerate(whereatses):
        whereatmat[i, :len(seq)] = seq
    goldmat = np.zeros((dims[0], dims[1]), dtype="int64")
    for i, seq in enumerate(golds):
        goldmat[i, :len(seq)] = seq
    return mat, whereatmat, goldmat


class SketchyDecoder(torch.nn.Module):
    def __init__(self, emb, cell, outlin, outline_role, ctx_enc, **kw):
        super(SketchyDecoder, self).__init__(**kw)
        self.ctx_enc = ctx_enc
        self.emb = emb
        self.cell = cell
        self.outlin = outlin
        self.outlin_role = outline_role
        self.sm = torch.nn.Softmax(1)

    def get_train_module(self):
        return SketchyDecoderTrain(self)

    def get_pred_module(self):
        return SketchyDecoderFree(self)


class SketchyDecoderTrain(torch.nn.Module):

    def __init__(self, skdec):
        super(SketchyDecoderTrain, self).__init__()
        self.d = skdec

    def forward(self, ctx, x, whereat):
        """
        :param ctx:  (batsize, inp_len) - context ^ids
        :param x:    (batsize, len) ^ids partial tree decoded
        :param whereat: (batsize,)  - where to insert prediction
        :return:
        """
        ctx_enc, ctx_mask = self.d.ctx_enc(ctx)

        x_emb, x_mask = self.d.emb(x)

        z = self.d.cell(x_emb, whereat, mask=x_mask, ctx=ctx_enc, ctx_mask = ctx_mask)

        outprobs = self.d.outlin(z)
        outprobs_role = self.d.outlin(z)

        outprobs = self.d.sm(outprobs)
        outprobs_role = self.d.sm(outprobs_role)

        return outprobs, outprobs_role


class SketchyDecoderFree(torch.nn.Module):
    def __init__(self, skdec):
        super(SketchyDecoderFree, self).__init__()
        self.d = skdec

    def forward(self, ctx, x):
        # TODO: implement free-running self-sampling decoding
        pass


def tst_bflr_seq_builder():
    xs =        [[1,2,3,4,5,6,0,0], [1,2,6]]
    levels =    [[0,1,2,2,1,0,0,0], [0,1,0]]
    rD = {0: "<MASK>", 1: "<START>", 6: "<END>", 2: "A", 3: "B", 4: "C", 5: "D"}
    D = {v: k for k, v in rD.items()}
    ret = get_bflr_sketchy_tf_seqs(xs, levels, D)
    return ret



if __name__ == "__main__":
    print(tst_bflr_seq_builder())
    # c = SketchyDecoderCell(8, 6)
    # x = torch.tensor([[1,2,3,0],
    #                   [2,0,0,0],
    #                   [5,4,3,2]]).long()
    # role_x = torch.tensor([[1,2,3,0],
    #                   [2,0,0,0],
    #                   [2,4,3,2]]).long()
    # D = {"<MASK>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    # role_D = {"<MASK>": 0, "*": 1, "NC": 2, "LS": 3, "NCLS": 4}
    #
    # emb = q.WordEmb(5, worddic=D)
    # role_emb = q.WordEmb(3, worddic=role_D)
    # emb = SketchyEmbedder(emb, role_emb)
    #
    # _x, _xmask = emb(x, role_x)
    # ctx = torch.randn(3, 7, 12)
    # where = torch.tensor([1,0,3])
    # z = c(_x, where, ctx=ctx, mask=_xmask)
