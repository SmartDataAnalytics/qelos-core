import torch


# region new attention
class Attention(torch.nn.Module):
    """ New attention base"""

    def __init__(self, **kw):
        super(Attention, self).__init__(**kw)

    def normalize(self, attention_scores, ctx_mask=None):
        raise NotImplemented("use subclass")

    def compute_scores(self, qry, ctx, ctx_mask=None):
        raise NotImplemented("use subclass")

    def summarize(self, norm_att_scores, ctx, values=None):
        raise NotImplemented("use subclass")

    def forward(self, qry, ctx, ctx_mask=None, values=None):
        scores = self.compute_scores(qry, ctx, ctx_mask=ctx_mask)
        alphas = self.normalize(scores, ctx_mask=ctx_mask)
        summary = self.summarize(alphas, ctx if values is None else values)
        return alphas, summary, scores


class AttentionScoreComputer(object):
    def compute_scores(self, qry, ctx, ctx_mask=None):
        raise NotImplemented("use subclass")


class DotScoreComputer(AttentionScoreComputer):
    def compute_scores(self, qry, ctx, ctx_mask=None):
        scores = torch.bmm(ctx, qry.unsqueeze(2)).squeeze(2)
        scores = scores + (torch.log(ctx_mask.float()) if ctx_mask is not None else 0)
        return scores


class GeneralDotScoreComputer(AttentionScoreComputer):
    def __init__(self, ctxdim=None, qdim=None, **kw):
        super(GeneralDotScoreComputer, self).__init__(**kw)
        self.W = torch.nn.Parameter(torch.empty(ctxdim, qdim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.W)

    def compute_scores(self, qry, ctx, ctx_mask=None):
        scores = torch.einsum("bi,ij,bkj->bk", qry, self.W, ctx)
        scores = scores + (torch.log(ctx_mask.float()) if ctx_mask is not None else 0)
        return scores


class FeedForwardScoreComputer(AttentionScoreComputer):
    def __init__(self, ctxdim=None, qdim=None, attdim=None, nonlin=torch.nn.Tanh(), **kw):
        super(FeedForwardScoreComputer, self).__init__(**kw)
        self.linear = torch.nn.Linear(ctxdim + qdim, attdim)
        self.nonlin = nonlin
        self.afterlinear = torch.nn.Linear(attdim, 1)

    def compute_scores(self, qry, ctx, ctx_mask=None):
        qry = qry.unsqueeze(1).repeat(1, ctx.size(1), 1)
        x = torch.cat([ctx, qry], 2)
        y = self.linear(x)  # (batsize, seqlen, attdim)
        y = self.nonlin(y)
        scores = self.afterlinear(y).squeeze(2)
        scores = scores + (torch.log(ctx_mask.float()) if ctx_mask is not None else 0)
        return scores


class FeedForwardMultScoreComputer(AttentionScoreComputer):
    def __init__(self, indim=None, attdim=None, nonlin=torch.nn.Tanh(), **kw):
        super(FeedForwardMultScoreComputer, self).__init__(**kw)
        self.linear = torch.nn.Linear(indim * 3, attdim)
        self.nonlin = nonlin
        self.afterlinear = torch.nn.Linear(attdim, 1)

    def compute_scores(self, qry, ctx, ctx_mask=None):
        qry = qry.unsqueeze(1).repeat(1, ctx.size(1), 1)
        x = torch.cat([ctx, qry, ctx * qry], 2)
        y = self.linear(x)  # (batsize, seqlen, attdim)
        y = self.nonlin(y)
        scores = self.afterlinear(y).squeeze(2)
        scores = scores + (torch.log(ctx_mask.float()) if ctx_mask is not None else 0)
        return scores


class AttentionSummarizer(object):
    def summarize(self, alphas, values):
        raise NotImplemented("use subclass")


class AverageSummarizer(AttentionSummarizer):
    def summarize(self, alphas, values):
        ret = values * alphas.unsqueeze(2)
        ret = ret.sum(1)
        if (alphas.sum(1) != 1).any():  # if weights don't some up to one, normalize
            ret = ret / alphas.sum(1, keepdim=True)
        return ret


class AttentionNormalizer(object):
    def normalize(self, attention_scores, ctx_mask=None):
        raise NotImplemented("use subclass")


class SoftmaxNormalizer(AttentionNormalizer):
    def __init__(self, **kw):
        super(SoftmaxNormalizer, self).__init__(**kw)
        self.sm = torch.nn.Softmax(-1)

    def normalize(self, attention_scores, ctx_mask=None):
        return self.sm(attention_scores)


class SigmoidNormalizer(AttentionNormalizer):
    def __init__(self, **kw):
        super(SigmoidNormalizer, self).__init__(**kw)
        self.sigm = torch.nn.Sigmoid()

    def normalize(self, attention_scores, ctx_mask=None):
        return self.sigm(attention_scores)


# normal attentions
class DefaultAttBase_(AverageSummarizer, SoftmaxNormalizer, Attention): pass


class DotAtt(DotScoreComputer, DefaultAttBase_): pass


class GeneralDotAtt(GeneralDotScoreComputer, DefaultAttBase_): pass


class FwdAtt(FeedForwardScoreComputer, DefaultAttBase_): pass


class FwdMultAtt(FeedForwardMultScoreComputer, DefaultAttBase_): pass

# endregion