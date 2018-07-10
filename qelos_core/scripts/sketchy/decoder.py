import torch
import qelos_core as q


## SKETCHY DECODER aka Encody McDecodyFace

class SketchyDecoderCell(torch.nn.Module):
    def __init__(self, indim, *dims, **kw):
        super(SketchyDecoderCell, self).__init__(**kw)
        dims = (indim,) + dims
        self.dims = dims
        self.encoder = q.FastestLSTMEncoder(*self.dims, bidir=True)

    def forward(self, x, whereat, mask=None, ctx=None, ctx_mask=None):
        """
        :param x:       (batsize, out_seqlen, embdim) the partial tree so far, in the form of a sequence of embeddings
        :param whereat: (batsize,) integers, where to insert prediction in x
        :param ctx:     (batsize, in_seqlen, dim) the context to be used with attention
        :return:
        """
        if mask is None:
            mask = torch.ones_like(x[:, :, 0])
        y = self.encoder(x, mask=mask)
        whereat_sel = whereat.view(whereat.size(0), 1, 1).repeat(1, 1, y.size(2))
        z = y.gather(1, whereat_sel).squeeze(1)  # (batsize, dim)  # not completely correct but fine

        if ctx is not None:     #TODO do attention
            attweights = torch.bmm(ctx, z.unsqueeze(2)) # (inp_len, dim) x (dim,1) -> (inp_len, 1), batsize times
            if ctx_mask is not None:
                attweights = attweights + torch.log(ctx_mask.float())
            attweights = torch.nn.functional.softmax(attweights, 1)
            summaries = ctx * attweights
            summaries = summaries.sum(1)

            z = torch.cat([z, summaries], 1)

        return z


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


if __name__ == "__main__":
    c = SketchyDecoderCell(8, 6)
    x = torch.tensor([[1,2,3,0],
                      [2,0,0,0],
                      [5,4,3,2]]).long()
    role_x = torch.tensor([[1,2,3,0],
                      [2,0,0,0],
                      [2,4,3,2]]).long()
    D = {"<MASK>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    role_D = {"<MASK>": 0, "*": 1, "NC": 2, "LS": 3, "NCLS": 4}

    emb = q.WordEmb(5, worddic=D)
    role_emb = q.WordEmb(3, worddic=role_D)
    emb = SketchyEmbedder(emb, role_emb)

    _x, _xmask = emb(x, role_x)
    ctx = torch.randn(3, 7, 12)
    where = torch.tensor([1,0,3])
    z = c(_x, where, ctx=ctx, mask=_xmask)
