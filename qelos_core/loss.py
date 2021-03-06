import torch
import qelos_core as q
from torch import nn
import numpy as np
import re
import math

EPS = 1e-6


class Penalty(torch.nn.Module):
    """ Reference implementation only, doesn't take into account ignoremasks or other masks.
        To use penalties, must do gather_penalties() first and enable penalties
            by setting their weights to positive nonzero. """

    __pp_name__ = "P"

    def __init__(self, name=None, size_average=True, weight=0., **kw):
        super(Penalty, self).__init__(**kw)
        self.size_average = size_average
        self.acc = None
        self.name = name
        assert(weight >= 0)
        self.weight = weight

    def forward(self, x):
        self.add(x)
        return self.acc

    def add(self, x):
        if self.acc is None:
            self.acc = torch.zeros_like(x)
        self.acc = self.acc + x

    def get_value(self):
        ret = self.acc.sum()
        if self.size_average:
            ret = ret / self.acc.size(0)
        return ret

    def get_penalty(self):
        """ Get weighted penalty value. Used by trainer. """
        value = torch.tensor(0.)
        if self.weight > 0.:
            value = self.get_value() * self.weight
        return value

    def reset(self):
        self.acc = None

    def batch_reset(self):
        self.reset()


def gather_penalties(m, subtype=None, name=None):
    for module in m.modules():
        if isinstance(module, Penalty):
            yieldit = True
            if subtype is None or (isinstance(module, subtype)):
                yieldit = yieldit
            else:
                yieldit = False
            if name is None or (module.name is not None and re.match(name, module.name)):
                yieldit = yieldit
            else:
                yieldit = False

            if yieldit:
                yield module


# TODO: REWRITE PROPERLY IN QELOS-CORE
# TODO: mask has no place here, mask must be applied in prediction modules

class Loss(nn.Module):
    def __init__(self, size_average=True, _size_avg_ignore_mask=False, **kw):
        super(Loss, self).__init__(**kw)
        self.size_average = size_average
        self._size_avg_ignore_mask = _size_avg_ignore_mask

    def forward(self, x, gold, mask=None, _noagg=False, **kw):
        y, ignoremask = self._forward(x, gold, mask=mask, **kw)
        y = y.float()
        if _noagg:
            return y, ignoremask

        if ignoremask is not None:
            y = y * ignoremask.float().clamp(0, 1)      # ensure ignoremask is not higher than 1
        if ignoremask is not None and self._size_avg_ignore_mask:
            total = ignoremask.long().sum().item()
        else:
            total = y.size(0)

        # try:
        loss = y.sum()
        # except Exception as e:
        #     q.embed()
        if self.size_average:
            loss /= total
        return loss


class PairRankingLoss(Loss):
    def __init__(self, size_average=True, margin=None, scale=1., **kw):
        super(PairRankingLoss, self).__init__(size_average=size_average, **kw)
        self.margin = margin
        self.scale = scale

    def _forward(self, x, gold, **kw):
        """ x is the difference in scores. optionally, gold is margin
            if x.dim() == 1, assuming margin loss
            if x.dim() == 2, assuming hinge loss of the two separately
        """
        if self.margin is None:     # use gold as margins
            margin = gold
        else:
            margin = self.margin

        zeros = torch.zeros(x.size(0)).to(x.device)
        if x.dim() == 1:
            loss = torch.max(zeros, margin * self.scale - x)
        elif x.dim() == 2:
            assert(x.size(1) == 2)
            lossA = torch.max(zeros, margin * self.scale - x[:, 0])
            lossB = torch.max(zeros, margin * self.scale + x[:, 1])
            loss = lossA + lossB
        return loss, None


class LinearLoss(Loss):
    """ LinearLoss or NoLoss. Use this if model returns loss """
    def _forward(self, x, gold, **kw):
        """ x is minimized, gold is ignored (and should be None) """
        return x, None


class SelectedLinearLoss(Loss):
    """ Same as LinearLoss, but with selection from tuple of outputs from model (that specifies lossses)
        To be used to output multiple losses from the model/ select one model output as training loss
    """
    def __init__(self, which, size_average=True, **kw):
        super(SelectedLinearLoss, self).__init__(size_average=size_average, **kw)
        self.which = which

    def _forward(self, model_outs, gold, **kw):
        if q.issequence(model_outs):
            return model_outs[self.which], None
        else:
            assert(self.which == 0)
            return model_outs, None


class DiscreteLoss(Loss):
    """ Loss with ignore_index(es), provides default implementation of _get_ignore_mask """
    def __init__(self, size_average=True, ignore_index=None, **kw):
        super(DiscreteLoss, self).__init__(size_average=size_average, **kw)
        if ignore_index is not None:
            if not q.issequence(ignore_index):
                self.ignore_indices = [ignore_index]
        else:
            self.ignore_indices = None

    def _get_ignore_mask(self, gold):
        mask = None     # (batsize,)
        if self.ignore_indices is not None:
            for ignore in self.ignore_indices:
                mask_i = (gold != ignore)   # zero for ignored ones
                if mask is None:
                    mask = mask_i
                else:
                    mask = mask & mask_i
        return mask


class SeqLoss(nn.Module):       # TODO: take end of sequence token into account
    def __init__(self, time_agg="sum", **kw):
        super(SeqLoss, self).__init__(**kw)
        self.time_agg = time_agg

    def _forward(self, probs, gold, mask=None):     # (batsize, seqlen, dim), idx^(batsize, seqlen)
        if probs.size(1) > gold.size(1):
            probs = probs[:, :gold.size(1)]
        batsize, seqlen, vocsize = probs.size()
        x = probs.contiguous().view(batsize * seqlen, vocsize)
        try:
            y = gold.contiguous().view(batsize * seqlen)
        except Exception as e:
            print((batsize, seqlen, gold.size()))
        if mask is not None:
            mask = mask.contiguous().view(batsize * seqlen, -1)

        l, ignoremask = super(SeqLoss, self)._forward(x, y, mask=mask)

        l = l.view(batsize, seqlen)

        outmask = None
        if ignoremask is not None:
            ignoremask = ignoremask.view(batsize, seqlen)
            outmask = ignoremask.long().sum(1) > 0
            totals = ignoremask.float().sum(1)
        else:
            totals = torch.FloatTensor(l.size(0)).to(l.device)
            totals.fill_(l.size(1))

        if self.time_agg == "sum":
            ltotal = l.float().sum(1)
        elif self.time_agg == "avg":
            ltotal = l.float().sum(1)
            totals = totals.clamp(min=EPS)
            ltotal = ltotal / totals
        elif self.time_agg == "all":
            if ignoremask is not None:
                l = (l.byte() | ~ ignoremask)
            ltotal = l.float().sum(1)
            ltotal = ltotal == float(l.size(1))
        elif self.time_agg == "eqtotal":
            ltotal = l.float().sum(1)
            print("DEPRECATED for 'all'")
            ltotal = (ltotal == totals)
        elif self.time_agg == "allone":
            ltotal = l.float().sum(1)
            print("DEPRECATED for 'all'")
            ltotal = (ltotal == l.size(1))

        return ltotal, outmask


class NLLLoss(DiscreteLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=None, **kw):
        super(NLLLoss, self).__init__(size_average=size_average, ignore_index=ignore_index, **kw)
        self.weight = weight

    def _forward(self, x, gold, mask=None):     # (batsize, vocsize)
        # probs for masked elements must have been zero by softmax
        ignoremask = self._get_ignore_mask(gold)

        if mask is not None:
            x = x + torch.log(mask.float())

        logprobs = -torch.gather(x, 1, gold.unsqueeze(1)).squeeze(1)

        if self.weight is not None:
            weights = self.weight[gold]
            logprobs = logprobs * weights

        if ignoremask is not None:      # GATHER solution
            _logprobs_cpy = torch.zeros(logprobs.size()).to(logprobs.device)
            _c_logprobs = torch.stack([_logprobs_cpy, logprobs], 1)
            logprobs = _c_logprobs.gather(1, ignoremask.unsqueeze(1).long())

            # SELVEC solution:
            # selvec = torch.masked_select(logprobs, ignoremask)
            # _logprobs_bak = logprobs + 0.
            # logprobs.data.fill_(0.)
            # logprobs.masked_scatter_(ignoremask, selvec)

            # MUL solution creates NaNs if any original probs had inf for masked elements
            # logprobs = logprobs * ignoremask.float()
        # q.embed()
        return logprobs, ignoremask


def logsumexp(x, axis=-1):
    xmax, _ = torch.max(x, axis, keepdim=True)
    _x = x - xmax
    _x = torch.exp(_x)
    _x = torch.sum(_x, axis, keepdim=True)
    lse = xmax + torch.log(_x)
    return lse.squeeze(-1)


class SeqKLLoss(DiscreteLoss):
    """ Straight implementation of cross-entropy loss for sequence prediction.
        Same as Sequence cross-entropy if no label smoothing.
        To be used after torch.nn.Softmax() """
    def __init__(self, time_average=False, time_agg=None, weight=None, size_average=True, ignore_index=None, label_smoothing=0., smooth_mix=0., mode="probs", **kw):
        """

        :param time_agg:        aggregation over time: if "avg", then averages, "sum" sums. Takes priority over time_average
        :param time_average:    averages over time if True. Default False.
        :param weight:          ?
        :param size_average:    average over batch (True) or sum (False)
        :param ignore_index:    which tokens in gold to ignore (mask)
        :param label_smoothing: how much uniform label smoothing to perform (between 0 and 1) to get target distribution
        :param smooth_mix:      how much to mix predictive distribution with target distribution
        :param mode:            "probs" (probs must be normalized by Softmax()), "logits" (probs are logits), "logprobs" (probs are log probs, produced by LogSoftmax())
        :param kw:
        """
        super(SeqKLLoss, self).__init__(size_average=size_average, ignore_index=ignore_index, **kw)
        if time_agg is None:
            time_agg = "avg" if time_average else "sum"
        assert(time_agg in "sum avg".split())
        self.time_agg = time_agg
        self.label_smoothing = label_smoothing
        self.smooth_mix = smooth_mix
        self.mode = mode

    def _forward(self, probs, gold, mask=None):
        if q.v(self.label_smoothing) > 0. or q.v(self.smooth_mix) > 0.:
            return self._forward_smooth(probs, gold, mask=mask)
        else:
            return self._forward_normal(probs, gold, mask=mask)

    def _forward_smooth(self, probs, gold, mask=None):
        if self.mode != "probs":
            raise NotImplemented("'logits' and 'logprobs' mode not implemented with softened targets (TODO)")

        if probs.size(1) > gold.size(1):
            probs = probs[:, :gold.size(1)]
        batsize, seqlen, vocsize = probs.size()

        ignoremask = self._get_ignore_mask(gold)        # whether to ignore a certain time step of a certain example
        outignoremask = None

        if mask is not None:
            probs = probs * mask

        prob_mask = (probs > 0).float()    # (batsize, seqlen, vocsize)
        if isinstance(q.v(self.label_smoothing), float):
            lsv = q.v(self.label_smoothing)
            assert(lsv >= 0 and lsv <= 1)
            prob_mask_weights = lsv / prob_mask.sum(2)
            _gold = torch.ones_like(probs) * prob_mask_weights.unsqueeze(2) * prob_mask     # masked uniform
            _gold.scatter_(2, gold.unsqueeze(2), (1 - lsv) + prob_mask_weights.unsqueeze(2))
        else:
            _gold = self.label_smoothing(gold, prob_mask)

        if q.v(self.smooth_mix) > 0.:
            smv = q.v(self.smooth_mix)
            _gold = _gold * (1 - smv) + smv * probs.detach()

        assert(np.allclose(_gold.sum(2).cpu().detach().numpy(),
                           np.ones((_gold.size(0), _gold.size(1))), atol=1e-3))

        log_probs = - (torch.log(probs + (1 - prob_mask)) - torch.log(_gold + (1 - prob_mask)))
        # REMARK: (1 - prob_mask) is added before log() to ensure that no -inf's are there
        kls = log_probs * _gold
        kls = kls * prob_mask       # prob can be removed
        gold_log_probs = kls.sum(2)

        seqlens = torch.tensor(seqlen).float().to(gold.device)

        if ignoremask is not None:
            gold_log_probs = gold_log_probs * ignoremask.float()        # should work because normal softmax was used --> no infs
            seqlens = ignoremask.float().sum(1)
            outignoremask = ignoremask.long().sum(1) > 0

        gold_log_probs = gold_log_probs.sum(1)
        if self.time_agg == "avg":
            gold_log_probs = gold_log_probs / seqlens.clamp(min=EPS)

        return gold_log_probs, outignoremask

    def _forward_normal(self, probs, gold, mask=None):
        if probs.size(1) > gold.size(1):        # if probs is longer than gold seq
            probs = probs[:, :gold.size(1)]
        batsize, seqlen, vocsize = probs.size()

        ignoremask = self._get_ignore_mask(gold)
        outignoremask = None

        if mask is not None:
            if self.mode == "probs":
                probs = probs * mask
            elif self.mode == "logits":
                probs = probs + torch.log(mask.float())
            elif self.mode == "logprobs":
                raise NotImplemented("mask in logprobs not implemented")

        gold_probs = probs.gather(2, gold.unsqueeze(2))
        assert(gold_probs.size(2) == 1)
        gold_probs = gold_probs.squeeze(2)

        if self.mode == "probs":
            gold_log_probs = - torch.log(gold_probs.clamp(min=1e-9))
        elif self.mode == "logprobs":
            gold_log_probs = - gold_probs
        elif self.mode == "logits":
            gold_log_probs = - gold_probs + logsumexp(probs)

        seqlens = torch.tensor(seqlen).float().to(gold.device)

        if ignoremask is not None:
            gold_log_probs = gold_log_probs * ignoremask.float()        # should work because normal softmax was used --> no infs
            seqlens = ignoremask.float().sum(1)
            outignoremask = ignoremask.long().sum(1) > 0

        gold_log_probs = gold_log_probs.sum(1)
        if self.time_agg == "avg":
            gold_log_probs = gold_log_probs / seqlens.clamp(min=EPS)

        return gold_log_probs, outignoremask


class SeqPPL_Loss(SeqKLLoss):
    def post_agg_epoch(self, x):
        """ function applied after aggregation (avg/sum) over whole epoch (so far) """
        return math.exp(x)


class SeqNLLLoss(SeqLoss, NLLLoss):
    def __init__(self, size_average=True, time_average=False, weight=None, ignore_index=None, **kw):
        super(SeqNLLLoss, self).__init__(size_average=size_average, time_agg="avg" if time_average else "sum",
                                         weight=weight, ignore_index=ignore_index, **kw)


class CrossEntropyLoss(NLLLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=None, temperature=1., **kw):
        super(CrossEntropyLoss, self).__init__(size_average=size_average, ignore_index=ignore_index, weight=weight, **kw)
        self.softmax = q.LogSoftmax(temperature=temperature)

    def _forward(self, scores, gold, mask=None):
        # softmax zeroes/mininfinites the masked symbols
        probs = self.softmax(scores, mask=mask)
        if isinstance(probs, tuple):
            probs = probs[0]
        logprobs, ignoremask = super(CrossEntropyLoss, self)._forward(probs, gold, mask=mask)
        return logprobs, ignoremask


class SeqCrossEntropyLoss(SeqLoss, CrossEntropyLoss):
    def __init__(self, size_average=True, time_average=False, weight=None, ignore_index=None, temperature=1., **kw):
        super(SeqCrossEntropyLoss, self).__init__(size_average=size_average, time_agg="avg" if time_average else "sum",
                                         weight=weight, ignore_index=ignore_index, temperature=temperature, **kw)


class RankingLoss(DiscreteLoss):
    def __init__(self, size_average=True, ignore_index=None,
                 negmode="random",
                 margin=None, ignore_below_margin=True, **kw):
        super(RankingLoss, self).__init__(size_average=size_average, ignore_index=ignore_index,
                                          **kw)
        self.margin = margin
        self.ignore_below_margin = ignore_below_margin
        self.negmode = negmode      # "random" or "best" or "negall"
        self._average_negall = True

    def _forward(self, scores, gold, mask=None):    # (batsize, numvoc), idx^(batsize,)
        # scores = scores - scores.min()
        goldscores = torch.gather(scores, 1, gold.unsqueeze(1)).squeeze()

        if mask is not None and mask[0, 1] > 1:
            mask = q.batchablesparse2densemask(mask)

        goldexamplemask = None

        if self.negmode == "random" or self.negmode == "negall":
            sampledist = scores.new(scores.size()).to(scores.device)
            sampledist.fill_(1.)
            sampledist.scatter_(1, gold.unsqueeze(1), 0)
            filtermask = scores > -np.infty
            if mask is not None:
                filtermask = filtermask & mask.byte()
            sampledist = sampledist * filtermask.float()
            sampledist_orig = sampledist
            if self.margin is not None and self.ignore_below_margin:
                cutoffs = goldscores - self.margin
                cutoffmask = scores > cutoffs.unsqueeze(1)
                sampledist = sampledist * cutoffmask.float()
            if (sampledist.sum(1) > 0).long().sum() < gold.size(0):
                # force to sample gold
                gold_onehot = torch.ByteTensor(sampledist.size()).to(sampledist.device)
                gold_onehot.fill_(0)
                gold_onehot.scatter_(1, gold.unsqueeze(1), 1)
                goldexamplemask = (sampledist.sum(1) != 0)
                # addtosampledist = sampledist_orig * examplemask.float().unsqueeze(1)
                addtosampledist = gold_onehot * (~goldexamplemask).unsqueeze(1)
                sampledist.masked_fill_(addtosampledist, 1)
            if self.negmode == "random":
                sample = torch.multinomial(sampledist, 1)
                negscores = torch.gather(scores, 1, sample).squeeze()
            elif self.negmode == "negall":
                negscores = scores * sampledist
                numnegs = sampledist.sum(1)
        elif self.negmode == "best":
            # scores = scores * mask.float() if mask else scores
            scores = scores + torch.log(mask.float()) if mask else scores
            bestscores, best = torch.max(scores, 1)
            secondscores = scores + 0
            secondscores.scatter_(1, best.unsqueeze(1), 0)
            secondbestscores, secondbest = torch.max(secondscores, 1)
            switchmask = best == gold
            sample = secondbest * switchmask.long() + best * (1 + (-1) * switchmask.long())
            negscores = secondbestscores * switchmask.float() + bestscores * (1 - switchmask.float())
            goldexamplemask = sample.squeeze() != gold
            # raise NotImplemented("some issues regarding implementation not resolved")
        else:
            raise q.SumTingWongException("unknown mode: {}".format(self.negmode))

        if self.negmode == "best" or self.negmode == "random":
            loss = negscores - goldscores
            if self.margin is not None:
                loss = torch.clamp(self.margin + loss, min=0)
            if goldexamplemask is not None:
                loss = goldexamplemask.float() * loss
        elif self.negmode == "negall":
            # negscores are 2D
            loss = negscores - goldscores.unsqueeze(1)
            if self.margin is not None:
                loss = torch.clamp(self.margin + loss, min=0)
            loss = loss * sampledist
            loss = loss.sum(1)
            if self._average_negall:
                loss = loss / numnegs
            if goldexamplemask is not None:
                loss = loss * goldexamplemask.float()

        ignoremask = self._get_ignore_mask(gold)
        if ignoremask is not None:
            loss = loss * ignoremask.float()

        return loss, ignoremask


class SeqRankingLoss(SeqLoss, RankingLoss):
    def __init__(self, size_average=True, time_average=False,
                 ignore_index=None, negmode="random",
                 margin=None, ignore_below_margin=True, **kw):
        super(SeqRankingLoss, self).__init__(size_average=size_average,
                                             time_agg="avg" if time_average else "sum",
                                             ignore_index=ignore_index,
                                             negmode=negmode,
                                             margin=margin,
                                             ignore_below_margin=ignore_below_margin,
                                             **kw)


class Accuracy(DiscreteLoss):
    def _forward(self, x, gold, mask=None):
        if mask is not None and mask[0, 1] > 1:     # batchable sparse
            mask = q.batchablesparse2densemask(mask)
        if mask is not None:
            x = x + torch.log(mask.float())
        ignoremask = self._get_ignore_mask(gold)
        maxes, best = torch.max(x, 1)
        same = best == gold
        if ignoremask is not None:
            same = same | ~ ignoremask
        return same.float(), ignoremask


class OldSeqAccuracy(SeqLoss, Accuracy):
    def __init__(self, size_average=True, ignore_index=None):
        super(OldSeqAccuracy, self).__init__(size_average=size_average,
                                          ignore_index=ignore_index,
                                          time_agg="all")


class SeqAccuracy(DiscreteLoss):
    """ very basic explicit seqaccuracy implementation.
        does not support batchable sparse mask """
    def _forward(self, x, gold, mask=None):
        """
        :param x: (batsize, seqlen, vocsize) - probabilities over output symbols for every time step
        :param gold: (batsize, seqlen) - ids of gold output symbols at every time step
        :param mask: (batsize, seqlen, vocsize) - optional mask to zero out probabilities over output symbols
        :return: loss value, ignormask
        """
        if mask is not None:
            x = x + torch.log(mask.float())
        ignoremask = self._get_ignore_mask(gold)
        _, best = torch.max(x, 2)       # (batsize, seqlen) - most probable symbols at every time step
        same = best == gold
        outignoremask = None
        if ignoremask is not None:
            same = same | ~ ignoremask   # set ignored portions to be same[i,j]=True
            outignoremask = ignoremask.long().sum(1) > 0
        sameseqs = same.long().sum(1)
        sameseqs = sameseqs == int(same.size(1))
        return sameseqs, outignoremask


class SeqElemAccuracy(DiscreteLoss):    # TODO take end of sequence token into account
    def forward(self, x, gold, mask=None):
        if x.size(1) > gold.size(1):
            x = x[:, :gold.size(1)]
        if mask is not None and mask[0, 0, 1] > 1:     # batchable sparse
            mask = q.batchablesparse2densemask(mask)
        if mask is not None:
            x = x + torch.log(mask.float())
        ignoremask = self._get_ignore_mask(gold)
        maxes, argmaxes = torch.max(x, dim=2)
        diff = argmaxes == gold
        if ignoremask is not None:
            diff = diff * ignoremask
            total = torch.sum(ignoremask.long()).item()
        else:
            total = gold.size(0) * gold.size(1)
        acc = torch.sum(diff.float())
        if self.size_average:
            acc = acc / total
        return acc, total


from nltk.translate.bleu_score import sentence_bleu
import warnings


class MacroBLEU(DiscreteLoss):      # TODO take end of sequence token into account
    """ macro-averaged BLEU over sequences """
    def __init__(self, order=4, predcut=None, ignore_index=None, **kw):
        """
        :param order:           n-gram order of BLEU
        :param predcut:         function to cut prediction. Gets the argmax over prediction and ignore_index kwarg.
                                Must fill all elements after end of sequence with provided ignore_index
        """
        super(MacroBLEU, self).__init__(ignore_index=ignore_index, **kw)
        self.order = order
        self.weights = tuple([1. / self.order for _ in range(self.order)])
        self.predcut = predcut
        warnings.filterwarnings("ignore", module="nltk")

    def forward(self, x, gold, mask=None):
        if x.size(1) > gold.size(1):
            x = x[:, :gold.size(1)]
        if mask is not None and mask[0, 0, 1] > 1:     # batchable sparse
            mask = q.batchablesparse2densemask(mask)
        if mask is not None:
            x = x + torch.log(mask.float())
        ignoremask = self._get_ignore_mask(gold)
        maxes, argmaxes = torch.max(x, dim=2)
        ignore_id = None
        if self.ignore_indices is not None:
            ignore_id = self.ignore_indices[0]
        argmaxes = argmaxes.cpu()
        if self.predcut is not None:
            argmaxes = self.predcut(argmaxes, ignore_index=ignore_id)
        gold = gold.cpu()
        bleus = 0.
        for i in range(gold.size(0)):
            predseq = [str(a) for a in list(argmaxes[i]) if a != ignore_id]
            goldseq = [str(a) for a in list(gold[i]) if a not in self.ignore_indices]
            bleu = sentence_bleu([goldseq], predseq, weights=self.weights)
            bleus += bleu

        total = gold.size(0)
        if self.size_average:
            bleus = bleus / total
        return bleus, total


class OldSeqNLLLoss(nn.NLLLoss):
    def __init__(self, weight=None, size_average=True, time_average=True, ignore_index=0):
        if ignore_index is not None:
            if not q.issequence(ignore_index):
                ignore_index = [ignore_index]
        else:
            ignore_index = None
        super(SeqNLLLoss, self).__init__(weight=weight, size_average=size_average, ignore_index=ignore_index)
        self.time_average = time_average

    def forward(self, probs, gold):
        """
        :param probs: (batsize, seqlen, vocsize) log-probabilities for each timestep
        :param gold:  (batsize, seqlen) correct values for each timestep
        :return:
        """
        batsize, seqlen, vocsize = probs.size()
        x = probs.view(batsize * seqlen, vocsize)
        y = gold.contiguous().view(batsize * seqlen)

        mask = None
        if self.ignore_index is not None:
            for ignore in self.ignore_index:
                mask_i = (y != ignore)   # zero for ignored ones
                if mask is None:
                    mask = mask_i
                else:
                    mask = mask * mask_i
            mask = mask.float()

        logprobs = -torch.gather(x, 1, y.unsqueeze(1)).squeeze()
        if self.weight is not None:
            weights = self.weight[y]
            logprobs = logprobs * weights

        if mask is not None:
            logprobs = logprobs * mask

        logprobs = logprobs.view(batsize, seqlen)
        if mask is not None:
            mask = mask.view(batsize, seqlen)
            totals = mask.sum(1).clamp(min=EPS)
        else:
            totals = logprobs.size(1)
        logprobsum = logprobs.sum(1)
        if self.time_average:
            logprobsum = logprobsum / totals
        t = logprobsum.size(0)

        loss = logprobsum.sum()
        if self.size_average:
            loss /= t
        return loss


class OldOldSeqAccuracy(nn.Module):
    def __init__(self, size_average=True, ignore_index=0):
        super(OldOldSeqAccuracy, self).__init__()
        self.size_average = size_average
        if ignore_index is not None:
            if not q.issequence(ignore_index):
                ignore_index = [ignore_index]
            self.ignore_index = ignore_index
        else:
            self.ignore_index = None
        self.EPS = 1e-6

    def forward(self, probs, gold, mask=None):     # (batsize, seqlen, vocsize), (batsize, seqlen)-idx
        mask = None
        if self.ignore_index is not None:
            for ignore in self.ignore_index:
                mask_i = (gold != ignore)   # zero for ignored ones
                if mask is None:
                    mask = mask_i
                else:
                    mask = mask * mask_i
        maxes, argmaxes = torch.max(probs, dim=2)
        diff = argmaxes != gold
        if mask is not None:
            diff = diff * mask
        diffsums = torch.sum(diff.long(), dim=1)
        total = gold.size(0)
        acc = torch.sum((diffsums == 0).long()).float()
        if self.size_average:
            acc = acc / total
        return acc




