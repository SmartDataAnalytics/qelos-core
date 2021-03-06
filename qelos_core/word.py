import math
from collections import OrderedDict

import numpy as np
import os, pickle as pkl
import json

import qelos_core as q
from torch import nn
import torch

EPS = 1e-6


class WordVecBase(object):
    masktoken = "<MASK>"
    raretoken = "<RARE>"

    def __init__(self, worddic, **kw):
        """
        Takes a worddic and provides word id lookup and vector retrieval interface
        """
        super(WordVecBase, self).__init__(**kw)
        self.D = OrderedDict() if worddic is None else worddic

    # region NON-BLOCK API :::::::::::::::::::::::::::::::::::::
    def getindex(self, word):
        return self.D[word] if word in self.D else self.D[self.raretoken] if self.raretoken in self.D else -1

    def __mul__(self, other):
        """ given word (string), returns index (integer) based on dictionary """
        return self.getindex(other)

    def __contains__(self, word):
        return word in self.D

    def getvector(self, word):
        raise NotImplemented()

    def __getitem__(self, word):
        """ given word (string or index), returns vector """
        return self.getvector(word)

    @property
    def shape(self):
        raise NotImplemented()

    def cosine(self, A, B):
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    def getdistance(self, A, B, distance=None):
        if distance is None:
            distance = self.cosine
        return distance(self.getvector(A), self.getvector(B))

    def __mod__(self, other):
        """ Given word (string or integer), returns vector.
            Does similarity computation if argument is a sequences. Compares first element of sequence with each of the following elements. """
        if isinstance(other, (tuple, list)):  # distance
            assert len(other) > 1
            if len(other) == 2:
                return self.getdistance(other[0], other[1])
            else:
                y = other[0]
                return [self.getdistance(y, x) for x in other[1:]]
        else:  # embed
            return self.__getitem__(other)
    # endregion


class WordEmbBase(WordVecBase, nn.Module):
    """
    All WordEmbs must be descendant.
    """
    def getvector(self, word):
        try:
            if q.isstring(word):
                word = self.D[word]
            wordid = torch.LongTensor([word])
            ret, _ = self(wordid)
            return ret.squeeze(0).detach().numpy()
        except Exception:
            return None

    def adapt(self, wdic):  # adapts to given word-idx dictionary
        """
        Adapts current word emb to a new dictionary
        """
        return AdaptedWordEmb(self, wdic)

    def override(self, wordemb,
                 which=None, whichnot=None):  # uses override vectors instead of base vectors if word in override dictionary
        """
        Overrides this wordemb's token vectors with the vectors from given wordemb.
        Optionally, restriction of which tokens to override can be specified by providing
            a list of tokens in which= argument.
        Optionally, exclusions can be made using whichnot
        """
        return OverriddenWordEmb(self, wordemb, which=which, whichnot=whichnot)

    def merge(self, wordemb, mode="sum"):
        """
        Merges this embedding with provided embedding using the provided mode.
        The dictionary of provided embedding must be identical to this embedding.
        """
        if not wordemb.D == self.D:
            raise q.SumTingWongException("must have identical dictionary")
        return MergedWordEmb(self, wordemb, mode=mode)


class ZeroWordEmb(WordEmbBase):
    def __init__(self, dim=50, worddic=None, **kw):
        super(ZeroWordEmb, self).__init__(worddic, **kw)
        self.dim = dim
        self.vecdim = dim
        # extract maskid and rareid from worddic
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        rareid = worddic[self.raretoken] if self.raretoken in worddic else None
        self.maskid = maskid

    def forward(self, x):
        outsize = x.size() + (self.dim,)
        zeros = torch.zeros(*outsize, device=x.device)
        mask = None
        if self.maskid is not None:
            mask = (x != self.maskid).int()
        return zeros, mask


class WordEmb(WordEmbBase):
    """ is a VectorEmbed with a dictionary to map words to ids """
    def __init__(self, dim=50, value=None, worddic=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, fixed=False, no_masking=False,
                 **kw):
        """
        Normal word embedder. Wraps nn.Embedding.

        :param dim: embedding vector dimension
        :param value: (optional) value to set the weight of nn.Embedding to
        :param worddic: worddic, must be provided
        :param max_norm: see nn.Embedding
        :param norm_type: see nn.Embedding
        :param scale_grad_by_freq: see nn.Embedding
        :param sparse: see nn.Embedding
        :param fixed: fixed embeddings
        :param no_masking: ignore usual mask id (default "<MASK>") in this instance of WordEmb 
            --> no masking (will return no mask), useful for using WordEmb in output vectors
        :param kw:
        """
        assert(worddic is not None)     # always needs a dictionary
        super(WordEmb, self).__init__(worddic, **kw)
        wdvals = list(worddic.values())
        assert(min(wdvals) >= 0)     # word ids must be positive

        # extract maskid and rareid from worddic
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        rareid = worddic[self.raretoken] if self.raretoken in worddic else None

        maskid = maskid if not no_masking else None

        self.maskid = maskid

        indim = max(worddic.values())+1        # to init from worddic
        self.embedding = nn.Embedding(indim, dim, padding_idx=maskid,
                                      max_norm=max_norm, norm_type=norm_type,
                                      scale_grad_by_freq=scale_grad_by_freq,
                                      sparse=sparse)
        if value is not None:
            self.embedding.weight = nn.Parameter(torch.from_numpy(value))
        if fixed is True:
            self.embedding.weight.requires_grad = False

        self.indim = indim
        self.outdim = dim
        self.vecdim = dim

        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        ret = self.embedding(x)
        mask = None
        if self.maskid is not None:
            mask = (x != self.maskid).int()
        return ret, mask


class AdaptedWordEmb(WordEmbBase):  # adapt to given dictionary, map extra words to rare
    def __init__(self, wordemb, wdic, **kw):
        D = wordemb.D
        # assert(wordemb.raretoken in D)     # must have rareid in D to map extra words to it
        super(AdaptedWordEmb, self).__init__(wdic, **kw)
        self.inner = wordemb

        rareid = D[wordemb.raretoken] if wordemb.raretoken in D else 0

        # maps all idx from wdic (new) to idx in wordemb.D (old)
        # maps words from wdic (new) that are missing in wordemb.D (old)
        #   to wordemb.D's rare id

        self.ad = {v: D[k] if k in D else rareid for k, v in wdic.items()}

        valval = np.ones((max(self.ad.keys()) + 1,), dtype="int64")
        for i in range(valval.shape[0]):
            valval[i] = self.ad[i] if i in self.ad else rareid
        self.adb = q.val(valval).v

    def forward(self, inp):
        # x = q.var(self.adb).cuda(inp).v.gather(0, inp)
        inpshape = inp.size()
        inp = inp.view(-1)
        x = self.adb.gather(0, inp)
        ret, msk = self.inner(x)
        if msk is not None:
            msk = msk.view(inpshape)
        ret = ret.view(*(inpshape+(-1,)))
        return ret, msk


class ComputedWordEmb(WordEmbBase):
    def __init__(self, data=None, computer=None, worddic=None):
        """
        Takes some numpy tensor, a module and a worddic and computes token vectors on the fly.

        :param data: numpy tensor, wrapped with tensor.from_numpy(), so must watch dtype
        :param computer: nn.Module that takes (some of) the data and computes a vector for each data row
        :param worddic: dictionary of tokens to ids
        """
        super(ComputedWordEmb, self).__init__(worddic=worddic)
        self.data = nn.Parameter(torch.from_numpy(data), requires_grad=False)
        self.computer = computer
        self.weight = None
        wdvals = list(worddic.values())
        assert(min(wdvals) >= 0)     # word ids must be positive

        # extract maskid and rareid from worddic
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        rareid = worddic[self.raretoken] if self.raretoken in worddic else None
        self.maskid = maskid
        # assert(maskid is None)
        # assert(rareid is None)
        self.indim = max(worddic.values())+1

    def forward(self, x):
        mask = None
        if self.maskid is not None:
            mask = x != self.maskid
        xshape = x.size()
        x = x.contiguous().view(-1)
        data = self.data.index_select(0, x)
        emb = self.computer(data)
        emb = emb.contiguous()
        emb = emb.view(*(xshape + (-1,)))
        return emb, mask


class OverriddenWordVecBase(WordVecBase, nn.Module):
    def __init__(self, base, override, which=None, whichnot=None, **kw):
        super(OverriddenWordVecBase, self).__init__(base.D)
        self.base = base
        self.over = override.adapt(base.D)
        self.vecdim = self.base.vecdim
        assert(not (which is not None and whichnot is not None))
        numout = max(base.D.values()) + 1
        whichnot = set()

        overridemask_val = np.zeros((numout,), dtype="float32")
        if which is None:   # which: list of words to override
            for k, v in base.D.items():     # for all symbols in base dic
                if k in override.D and k not in whichnot:         # if also in override dic
                    overridemask_val[v] = 1
        else:
            for k in which:
                if k in override.D:     # TODO: if k from which is missing from base.D
                    overridemask_val[base.D[k]] = 1
        self.overridemask = q.val(overridemask_val).v


class OverriddenWordEmb(OverriddenWordVecBase, WordEmbBase):
    def forward(self, x):
        x = x.contiguous()
        xshape = x.size()
        x = x.view(-1)
        base_emb, base_msk = self.base(x)
        over_emb, over_msk = self.over(x)
        over_msk_select = torch.gather(self.overridemask, 0, x)
        emb = base_emb * (1 - over_msk_select.unsqueeze(1)) + over_emb * over_msk_select.unsqueeze(1)
        emb = emb.view(*(xshape + (-1,)))
        msk = None
        if base_msk is not None:
            msk = base_msk.view(xshape)
        return emb, msk


class MergedWordVecBase(WordVecBase):
    def __init__(self, base, merge, mode="sum"):
        super(MergedWordVecBase, self).__init__(base.D)
        self.base = base
        self.merg = merge
        self.mode = mode
        if not mode in ("sum", "cat"):
            raise q.SumTingWongException("{} merge mode not suported".format(mode))


class MergedWordEmb(MergedWordVecBase, WordEmbBase):
    def forward(self, x):
        base_emb, base_msk = self.base(x)
        merg_emb, merg_msk = self.merg(x)
        if self.mode == "sum":
            emb = base_emb + merg_emb
            msk = base_msk      # since dictionaries are identical
        elif self.mode == "cat":
            emb = torch.cat([base_emb, merg_emb], 1)
            msk = base_msk
        else:
            raise q.SumTingWongException()
        return emb, msk


class PretrainedWordVec(object):
    defaultpath = "../data/glove/glove.%dd"
    masktoken = "<MASK>"
    raretoken = "<RARE>"

    trylowercase = True

    loadcache = {}
    useloadcache = True

    @classmethod
    def _get_path(cls, dim, path=None):
        # if dim=None, load all
        path = cls.defaultpath if path is None else path
        relpath = path % dim
        path = os.path.join(os.path.dirname(__file__), relpath)
        return path

    def loadvalue(self, path, dim, indim=None, worddic=None, maskid=True, rareid=True):
        # TODO: nonstandard mask and rareid?
        tt = q.ticktock(self.__class__.__name__)
        tt.tick()
        # load weights
        if path not in self.loadcache:
            W = np.load(path+".npy")
        else:
            W = self.loadcache[path][0]
        tt.tock("vectors loaded")

        # load words
        tt.tick()
        if path not in self.loadcache:
            words = json.load(open(path+".words"))
        else:
            words = self.loadcache[path][1]
        tt.tock("words loaded")

        # cache
        if self.useloadcache:
            self.loadcache[path] = (W, words)

        # select
        if indim is not None:
            W = W[:indim, :]

        if rareid:
            W = np.concatenate([np.zeros_like(W[0, :])[np.newaxis, :], W], axis=0)
        if maskid:
            W = np.concatenate([np.zeros_like(W[0, :])[np.newaxis, :], W], axis=0)

        tt.tick()

        # dictionary
        D = OrderedDict()
        i = 0
        if maskid is not None:
            D[self.masktoken] = i; i+=1
        if rareid is not None:
            D[self.raretoken] = i; i+=1
        wordset = set(words)
        for j, word in enumerate(words):
            if indim is not None and j >= indim:
                break
            if word.lower() not in wordset and self.trylowercase:
                word = word.lower()
            D[word] = i
            i += 1
        tt.tock("dictionary created")

        if worddic is not None:
            vocsize = max(worddic.values()) + 1
            new_weight = np.zeros((vocsize, W.shape[1]), dtype=W.dtype)
            new_dic = {}
            for k, v in worddic.items():
                if k in D:
                    new_weight[v, :] = W[D[k], :]
                    new_dic[k] = v

            W = new_weight
            D = new_dic

        return W, D


class PretrainedWordEmb(WordEmb, PretrainedWordVec):

    def __init__(self, dim, vocabsize=None, path=None,
                 worddic=None, fixed=True, incl_maskid=True,
                 incl_rareid=True, value=None, project=None, **kw):
        """
        WordEmb that sets the weight of nn.Embedder to loaded pretrained vectors.
        Adds a maskid and rareid as specified on the class.

        :param dim: token vector dimensions
        :param vocabsize: (optional) number of tokens to load
        :param path: (optional) where to load from.
                     Must be of format .../xxx%dxxx.
                     Files must be separated in .npy matrix and .words list.
                     Defaults to glove in qelos/data/.
        :param fixed: no learning
        :param incl_maskid: includes a <MASK> token in dictionary and assigns it id 0
        :param incl_rareid: includes a <RARE> token in dictionary and assigns it id 1 if incl_maskid was True, and id 0 otherwise
        """
        assert("worddic" not in kw)
        self.path = path
        self.dim = dim
        if value is None:
            path = self._get_path(dim, path=path)
            value, wdic = self.loadvalue(path, dim, indim=vocabsize,
                                         worddic=worddic, maskid=incl_maskid,
                                         rareid=incl_rareid)
        else:
            wdic = worddic
        self.allwords = list(wdic.keys())
        super(PretrainedWordEmb, self).__init__(dim=dim, value=value,
                                                worddic=wdic, fixed=fixed, **kw)

        self.project = project

    def forward(self, x):
        ret, mask = super(PretrainedWordEmb, self).forward(x)
        if self.project is not None:
            ret = self.project(ret)
        return ret, mask

    def subclone(self, worddic, fixed=True):
        vocsize = max(worddic.values()) + 1
        dim = self.embedding.weight.size(1)

        cloneweight = np.zeros((vocsize, dim), dtype="float32")
        clonedic = {}
        for k, v in worddic.items():
            if k in self.D:
                cloneweight[v] = self.embedding.weight[self.D[k]].detach().numpy()
                clonedic[k] = v

        ret = PretrainedWordEmb(cloneweight.shape[1],
                worddic=clonedic, fixed=fixed, value=cloneweight)

        return ret


class PartiallyPretrainedWordEmb(WordEmb, PretrainedWordVec):
    def __init__(self, dim=50, worddic=None, keepvanilla=None, path=None, gradfracs=(1., 1.), **kw):
        """
        :param dim:         embedding dimension
        :param worddic:     which words to create embeddings for, must map from strings to ids
        :param keepvanilla: set of words which will be kept in the vanilla set of vectors
                            even if they occur in pretrained embeddings
        :param path:        where to load pretrained word from
        :param gradfracs:   tuple (vanilla_frac, pretrained_frac)
        :param kw:
        """
        super(PartiallyPretrainedWordEmb, self).__init__(dim=dim, worddic=worddic, **kw)
        path = self._get_path(dim, path=path)
        value, wdic = self.loadvalue(path, dim, indim=None,
                                     worddic=None, maskid=None,
                                     rareid=None)
        value = torch.tensor(value)
        self.mixmask = q.val(np.zeros((len(self.D),), dtype="float32")).v

        for k, v in self.D.items():
            if k in wdic and (keepvanilla is None or k not in keepvanilla):
                self.embedding.weight[v, :] = value[wdic[k], :]
                self.mixmask[v] = 1

        self.embedding.weight = torch.nn.Parameter(self.embedding.weight)

        self.gradfrac_vanilla, self.gradfrac_pretrained = gradfracs

        def apply_gradfrac(grad):
            if self.gradfrac_vanilla != 1.:
                grad = grad * ((1 - self.mixmask.unsqueeze(1)) * q.v(self.gradfrac_vanilla)
                               + self.mixmask.unsqueeze(1))
            if self.gradfrac_pretrained != 1.:
                grad = grad * (self.mixmask.unsqueeze(1) * q.v(self.gradfrac_pretrained)
                               + (1 - self.mixmask.unsqueeze(1)))
            return grad

        self.embedding.weight.register_hook(apply_gradfrac)


class WordLinoutBase(WordVecBase, nn.Module):
    def __init__(self, worddic, **kw):
        super(WordLinoutBase, self).__init__(worddic, **kw)
        self.cosnorm = False

    def getvector(self, word):
        try:
            if q.isstring(word):
                word = self.D[word]
            wordid = torch.LongTensor([word])
            ret = self._getvector(wordid)
            return ret.squeeze().detach().numpy()
        except Exception as e:
            return None

    def _getvector(self, wordid):
        raise NotImplemented()

    def adapt(self, wdic):  # adapts to given word-idx dictionary
        return AdaptedWordLinout(self, wdic)

    def override(self, wordemb,
                 which=None):  # uses override vectors instead of base vectors if word in override dictionary
        wordemb.cosnorm = self.cosnorm
        ret = OverriddenWordLinout(self, wordemb, which=which)
        return ret

    def merge(self, x, mode="sum"):
        x.cosnorm = self.cosnorm
        if not self.D == x.D:
            raise q.SumTingWongException()
        return MergedWordLinout(self, x, mode=mode)


class ZeroWordLinout(WordLinoutBase):
    def __init__(self, indim, worddic=None, cosnorm=False):
        super(ZeroWordLinout, self).__init__(worddic)
        self.indim = indim
        self.vecdim = indim
        self.outdim = max(worddic.values()) + 1
        self.cosnorm = cosnorm

    def forward(self, x, _ret_cosnorm=False, **kw):
        outsize = x.size()[:-1] + (self.outdim,)
        zeros = torch.zeros(*outsize, device=x.device)
        if _ret_cosnorm:
            return zeros, torch.sum(zeros, 0).unsqueeze(0)
        return zeros


class WordLinout(WordLinoutBase):
    def __init__(self, indim, worddic=None, weight=None, set_bias=None, bias=True, fixed=False, cosnorm=False):
        """
        Linear block to be used at the output for computing scores over a vocabulary of tokens. Usually followed by Softmax.

        :param indim: incoming dimension
        :param worddic: dictionary of words to ids
        :param weight: (optional) custom weight matrix. Must be numpy array. Watch the dtype
        :param set_bias: (optional) custom bias. Must be numpy array. Watch the dtype.
        :param bias: (optional) use bias
        :param fixed: (optional) don't train this
        """
        super(WordLinout, self).__init__(worddic)
        wdvals = list(worddic.values())
        assert(min(wdvals) >= 0)     # word ids must be positive

        # extract maskid and rareid from worddic
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        rareid = worddic[self.raretoken] if self.raretoken in worddic else None
        self.maskid = maskid

        outdim = max(worddic.values())+1        # to init from worddic
        self.outdim = outdim
        self.indim = indim
        self.vecdim = indim

        self.cosnorm = cosnorm

        if cosnorm and bias:
            print("disabling bias because cosnorm")
            bias = False

        self.lin = nn.Linear(indim, outdim, bias=bias)

        self.reset_parameters()

        if weight is not None:
            self.lin.weight = nn.Parameter(torch.from_numpy(weight))
        if set_bias is not None:
            self.lin.bias = nn.Parameter(torch.from_numpy(bias))
        if fixed is True:
            self.lin.weight.requires_grad = False
            if bias is True:
                self.lin.bias.requires_grad = False

    def reset_parameters(self):
        initrange = 0.1
        if self.lin.bias is not None:
            self.lin.bias.data.zero_()
        self.lin.weight.data.uniform_(-initrange, initrange)

    def _getvector(self, wordid):
        vec = self.lin.weight.index_select(0, wordid)
        return vec

    def forward(self, x, mask=None, _do_cosnorm=False, _retcosnorm=False, _no_mask_log=False):
        ret = self.lin(x)
        if (self.cosnorm or _do_cosnorm) and not _retcosnorm:      # normalize cosnorm
            normweight = torch.norm(self.lin.weight, 2, 1).unsqueeze(0)
            normx = torch.norm(x, 2, 1)
            ret = ret / torch.clamp(normweight, min=EPS)
            ret = ret / torch.clamp(normx.unsqueeze(1), min=EPS)
        if mask is not None:
            if _no_mask_log is False:
                ret = ret + torch.log(mask.float())
            else:
                ret = ret * mask.float()
        if _retcosnorm:
            cosnorm = torch.pow(self.lin.weight, 2).sum(1).unsqueeze(0)
            return ret, cosnorm
        return ret#, mask ?


class ComputedWordLinout(WordLinoutBase):
    def __init__(self, data=None, computer=None, worddic=None, bias=False, cosnorm=False):
        """
        WordLinout that computes the weight matrix of the Linear transformation dynamically
        based on provided data and computer.
        :param data:    numpy array, 2D or more, one symbol data per row. Automatically wrapped so watch the dtype
        :param computer: module that builds vectors for rows of data
        :param worddic: token dictionary from token to id
        :param bias: (optional) use bias (not computed)
        """
        super(ComputedWordLinout, self).__init__(worddic)
        self.data = q.val(torch.tensor(data)).v
        self.computer = computer
        # TODO: batches for computer???

        wdvals = list(worddic.values())
        assert(min(wdvals) >= 0)     # word ids must be positive

        # extract maskid and rareid from worddic
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        rareid = worddic[self.raretoken] if self.raretoken in worddic else None

        self.outdim = max(worddic.values())+1
        self.cosnorm = cosnorm
        if cosnorm and bias:
            print("disabling bias because cosnorm")
            bias = False
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.outdim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.base_weight = None     # zero weight

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.uniform_(-stdv, stdv)

    def forward(self, x, mask=None, _do_cosnorm=False, _retcosnorm=False, _no_mask_log=False):        # (batsize, indim), (batsize, outdim)
        if mask is not None:
            mask = mask.long()
            # select data, compute vectors, build switcher
            msk = mask.sum(0)       # --> (outdim,)
            msk = (msk > 0).long()
            compute_ids = msk.nonzero()    # which ids to compute
            if compute_ids.size(0) > 0:    # not all zeros
                compute_ids = compute_ids.squeeze(1)
                data_select = self.data[compute_ids]
                comp_weight = self.computer(data_select)        # (num_data_select, indim)
                comp_weight = comp_weight.contiguous()
                indim = comp_weight.size(1)
                # if self.base_weight is None or self.base_weight.size(1) != indim:
                #     self.base_weight = q.var(torch.zeros(1, indim)).cuda(x).v
                base_weight = torch.zeros(1, indim, device=x.device)
                weight = torch.cat([base_weight, comp_weight], 0)      # prepend a zero vector for masked ones
                index_transform = (torch.cumsum(msk, 0) * msk).long()
                weight = weight.index_select(0, index_transform)
            else:
                data_select = self.data[0:1]        # this computing is done to get dimensions
                comp_weight = self.computer(data_select)        # (num_data_select, indim)
                comp_weight = comp_weight.contiguous()
                indim = comp_weight.size(1)
                weight = torch.zeros(mask.size(1), indim, device=x.device)
        else:
            weight = self.computer(self.data)
            weight = weight.contiguous()
        out = torch.mm(x, weight.t())
        if (self.cosnorm or _do_cosnorm) and not _retcosnorm:
            normweight = torch.norm(weight, 2, 1)
            normx = torch.norm(x, 2, 1)
            out = out / torch.clamp(normweight, min=EPS).unsqueeze(0)
            out = out / torch.clamp(normx, min=EPS).unsqueeze(1)
        if self.bias:
            bias = self.bias if mask is not None else self.bias * mask
            out += bias
        if mask is not None:
            if _no_mask_log:
                out = out * mask.float()
            else:
                out = out + torch.log(mask.float())
        if _retcosnorm:
            cosnorm = torch.sum(torch.pow(weight, 2), 1).unsqueeze(0)
            return out, cosnorm
        return out#, mask ?


class PretrainedWordLinout(WordLinout, PretrainedWordVec):
    def __init__(self, dim, vocabsize=None, path=None, worddic=None, fixed=True,
                 incl_maskid=True, incl_rareid=True, bias=False,
                 cosnorm=False,
                 **kw):
        """
        WordLinout that sets the weight of the contained nn.Linear to loaded pretrained vectors.
        Adds a maskid and rareid as specified on the class.

        :param dim: token vector dimensions
        :param vocabsize: (optional) number of tokens to load
        :param path: (optional) where to load from.
                     Must be of format .../xxx%dxxx.
                     Files must be separated in .npy matrix and .words list.
                     Defaults to glove in qelos/data/.
        :param fixed: (optional) no learning. Disables bias automatically.
        :param incl_maskid: (optional) includes a <MASK> token in dictionary and assigns it id 0
        :param incl_rareid: (optional) includes a <RARE> token in dictionary and assigns it id 1 if incl_maskid was True, and id 0 otherwise
        :param bias: (optional) use bias. Default initial bias value is random (bias disabled when fixed=True).
        """
        assert ("worddic" not in kw)
        path = self._get_path(dim, path=path)
        value, wdic = self.loadvalue(path, dim, indim=vocabsize, worddic=worddic, maskid=incl_maskid, rareid=incl_rareid)
        self.allwords = list(wdic.keys())
        bias = bias and not fixed
        super(PretrainedWordLinout, self).__init__(dim, weight=value,
                                                   worddic=wdic, fixed=fixed,
                                                   bias=bias, cosnorm=cosnorm,
                                                   **kw)


class AdaptedWordLinout(WordLinoutBase):
    def __init__(self, wordlinout, wdic, **kw):
        D = wordlinout.D
        # assert (self.raretoken in D)  # must have rareid in D to map extra words to it
        # assert(wordlinout.raretoken in wdic)
        super(AdaptedWordLinout, self).__init__(wdic, **kw)
        self.inner = wordlinout

        self.cosnorm = wordlinout.cosnorm

        rareid_new2old = D[wordlinout.raretoken] if wordlinout.raretoken in D else 0
        rareid_old2new = wdic[self.raretoken] if self.raretoken in wdic else 0

        self.new_to_old_d = {v: D[k] if k in D else rareid_new2old
                   for k, v in wdic.items()}
        # mapping from new indexes (wdic) to old indexes (wordlinout)
        self.old_to_new_d = {v: wdic[k] if k in wdic else rareid_old2new
                             for k, v in D.items()}
        # mapping from old indexes (wordlinout) to new indexes (wdic)

        numnew = max(self.new_to_old_d.keys()) + 1
        numold = max(self.old_to_new_d.keys()) + 1

        new_to_old = np.zeros((numnew,), dtype="int64")
        for i in range(new_to_old.shape[0]):
            j = self.new_to_old_d[i] if i in self.new_to_old_d else rareid_new2old
            new_to_old[i] = j
        self.new_to_old = q.val(new_to_old).v  # for every new dic word id, contains old dic id
        # index in new dic contains idx value of old dic
        # --> used to slice from matrix in old idxs to get matrix in new idxs

        old_to_new = np.zeros((numold,), dtype="int64")
        for i in range(old_to_new.shape[0]):
            j = self.old_to_new_d[i] if i in self.old_to_new_d else rareid_old2new
            old_to_new[i] = j
        self.old_to_new = q.val(old_to_new).v  # for every old dic word id, contains new dic id

    def _getvector(self, wordid):
        wordid = self.new_to_old[wordid]
        return self.inner.lin.weight[wordid]

    def forward(self, x, mask=None, _do_cosnorm=False, _retcosnorm=False, _no_mask_log=False):       # (batsize, indim), (batsize, outdim)
        innermask = mask.index_select(1, self.old_to_new) if mask is not None else None
        # TODO: SOMETHING WRONG, innermask is all zero
        baseout = self.inner(x, mask=innermask, _do_cosnorm=_do_cosnorm, _retcosnorm=_retcosnorm, _no_mask_log=_no_mask_log)     # (batsize, outdim) --> need to permute columns
        if _retcosnorm:
            baseout, cosnorm = baseout
        out = baseout.index_select(1, self.new_to_old)
        if _retcosnorm:
            cosnorm = cosnorm.index_select(1, self.new_to_old)
            return out, cosnorm
        return out#, mask?


class OverriddenWordLinout(OverriddenWordVecBase, WordLinoutBase):
    def forward(self, x, mask=None, _do_cosnorm=False, _retcosnorm=False, _no_mask_log=False):    # (batsize, indim), (batsize, outdim)
        baseres = self.base(x, mask=mask, _do_cosnorm=_do_cosnorm, _retcosnorm=_retcosnorm, _no_mask_log=True)
        overres = self.over(x, mask=mask, _do_cosnorm=_do_cosnorm, _retcosnorm=_retcosnorm, _no_mask_log=True)
        if _retcosnorm:
            baseres, basecosnorm = baseres
            overres, overcosnorm = overres
        res = self.overridemask.unsqueeze(0) * overres \
              + (1 - self.overridemask.unsqueeze(0)) * baseres
        if mask is not None:
            if _no_mask_log:
                res = res * mask.float()
            else:
                res = res + torch.log(mask.float())
        if _retcosnorm:
            cosnorm = self.overridemask.unsqueeze(0) * overcosnorm \
                + (1 - self.overridemask.unsqueeze(0)) * basecosnorm
            return res, cosnorm
        return res#, mask


class MergedWordLinout(MergedWordVecBase, WordLinoutBase):
    def forward(self, x, mask=None, _do_cosnorm=False, _retcosnorm=False, _no_mask_log=False):
        if self.mode == "cat":      # need to split up input
            basex = x[:, :self.base.vecdim]
            mergx = x[:, self.base.vecdim:]
            # TODO: not all wordlinouts have .vecdim
        elif self.mode == "sum":
            basex, mergx = x, x
        else:
            raise q.SumTingWongException()
        baseres = self.base(basex, mask=mask, _do_cosnorm=False,
                _retcosnorm=_retcosnorm or self.cosnorm or _do_cosnorm,
                            _no_mask_log=_no_mask_log)
        mergres = self.merg(mergx, mask=mask, _do_cosnorm=False,
                _retcosnorm=_retcosnorm or self.cosnorm or _do_cosnorm,
                            _no_mask_log=_no_mask_log)
        if _retcosnorm or self.cosnorm or _do_cosnorm:
            baseres, basecosnorm = baseres
            mergres, mergcosnorm = mergres
            cosnorm = basecosnorm + mergcosnorm
        res = baseres + mergres
        if _retcosnorm:
            return res, cosnorm
        if self.cosnorm or _do_cosnorm:
            res = res / torch.clamp(torch.norm(x, 2, 1).unsqueeze(1), min=EPS)
            res = res / torch.clamp(cosnorm, min=EPS).pow(1./2)
        return res