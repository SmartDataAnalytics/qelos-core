from __future__ import print_function
import argparse
import collections
import inspect
import re
import os
import signal
import sys
from datetime import datetime as dt
import pickle
import nltk

import numpy as np
import unidecode
from IPython import embed
import qelos_core as q
import torch
from torch.utils.data.dataset import Dataset
# torch-independent utils


class StringMatrix():
    protectedwords = ["<MASK>", "<RARE>", "<START>", "<END>"]

    def __init__(self, maxlen=None, freqcutoff=0, topnwords=None, indicate_start_end=False, indicate_start=False, indicate_end=False):
        self._strings = []
        self._wordcounts_original = dict(zip(self.protectedwords, [0] * len(self.protectedwords)))
        self._dictionary = dict(zip(self.protectedwords, range(len(self.protectedwords))))
        self._dictionary_external = False
        self._rd = None
        self._next_available_id = len(self._dictionary)
        self._maxlen = 0
        self._matrix = None
        self._max_allowable_length = maxlen
        self._rarefreq = freqcutoff
        self._topnwords = topnwords
        self._indic_e, self._indic_s = False, False
        if indicate_start_end:
            self._indic_s, self._indic_e = True, True
        if indicate_start:
            self._indic_s = indicate_start
        if indicate_end:
            self._indic_e = indicate_end
        self._rarewords = set()
        self.tokenize = tokenize
        self._cache_p = None

    def clone(self):
        n = StringMatrix()
        n.tokenize = self.tokenize
        if self._matrix is not None:
            n._matrix = self._matrix.copy()
            n._dictionary = self._dictionary.copy()
            n._rd = self._rd.copy()

        n._strings = self._strings
        return n

    def __len__(self):
        if self._matrix is None:
            return len(self._strings)
        else:
            return self.matrix.shape[0]

    def cached(self, p):
        self._cache_p = p
        if os.path.isfile(p):
            pickle.load()

    def __getitem__(self, item, *args):
        if self._matrix is None:
            return self._strings[item]
        else:
            ret = self.matrix[item]
            if len(args) == 1:
                ret = ret[args[0]]
            ret = self.pp(ret)
            return ret

    @property
    def numwords(self):
        return len(self._dictionary)

    @property
    def numrare(self):
        return len(self._rarewords)

    @property
    def matrix(self):
        if self._matrix is None:
            raise Exception("finalize first")
        return self._matrix

    @property
    def D(self):
        return self._dictionary

    def set_dictionary(self, d):
        """ dictionary set in this way is not allowed to grow,
        tokens missing from provided dictionary will be replaced with <RARE>
        provided dictionary must contain <RARE> if missing tokens are to be supported"""
        print("setting dictionary")
        self._dictionary_external = True
        self._dictionary = {}
        self._dictionary.update(d)
        self._next_available_id = max(self._dictionary.values()) + 1
        self._wordcounts_original = dict(zip(list(self._dictionary.keys()), [0]*len(self._dictionary)))
        self._rd = {v: k for k, v in self._dictionary.items()}

    @property
    def RD(self):
        return self._rd

    def d(self, x):
        return self._dictionary[x]

    def rd(self, x):
        return self._rd[x]

    def pp(self, matorvec):
        def pp_vec(vec):
            return " ".join([self.rd(x) if x in self._rd else "<UNK>" for x in vec if x != self.d("<MASK>")])
        ret = []
        if matorvec.ndim == 2:
            for vec in matorvec:
                ret.append(pp_vec(vec))
        else:
            return pp_vec(matorvec)
        return ret

    def add(self, x):
        tokens = self.tokenize(x)
        tokens = tokens[:self._max_allowable_length]
        if self._indic_s is not False and self._indic_s is not None:
            indic_s_sym = "<START>" if not isstring(self._indic_s) else self._indic_s
            tokens = [indic_s_sym] + tokens
        if self._indic_e is not False and self._indic_e is not None:
            indic_e_sym = "<END>" if not isstring(self._indic_e) else self._indic_e
            tokens = tokens + [indic_e_sym]
        self._maxlen = max(self._maxlen, len(tokens))
        tokenidxs = []
        for token in tokens:
            if token not in self._dictionary:
                if not self._dictionary_external:
                    self._dictionary[token] = self._next_available_id
                    self._next_available_id += 1
                    self._wordcounts_original[token] = 0
                else:
                    assert("<RARE>" in self._dictionary)
                    token = "<RARE>"    # replace tokens missing from external D with <RARE>
            self._wordcounts_original[token] += 1
            tokenidxs.append(self._dictionary[token])
        self._strings.append(tokenidxs)
        return len(self._strings)-1

    def finalize(self):
        ret = np.zeros((len(self._strings), self._maxlen), dtype="int64")
        for i, string in enumerate(self._strings):
            ret[i, :len(string)] = string
        self._matrix = ret
        self._do_rare_sorted()
        self._rd = {v: k for k, v in self._dictionary.items()}
        self._strings = None

    def _do_rare_sorted(self):
        """ if dictionary is not external, sorts dictionary by counts and applies rare frequency and dictionary is changed """
        if not self._dictionary_external:
            sortedwordidxs = [self.d(x) for x in self.protectedwords] + \
                             ([self.d(x) for x, y
                              in sorted(self._wordcounts_original.items(), key=lambda (x, y): y, reverse=True)
                              if y >= self._rarefreq and x not in self.protectedwords][:self._topnwords])
            transdic = zip(sortedwordidxs, range(len(sortedwordidxs)))
            transdic = dict(transdic)
            self._rarewords = {x for x in self._dictionary.keys() if self.d(x) not in transdic}
            rarewords = {self.d(x) for x in self._rarewords}
            self._numrare = len(rarewords)
            transdic.update(dict(zip(rarewords, [self.d("<RARE>")]*len(rarewords))))
            # translate matrix
            self._matrix = np.vectorize(lambda x: transdic[x])(self._matrix)
            # change dictionary
            self._dictionary = {k: transdic[v] for k, v in self._dictionary.items() if self.d(k) in sortedwordidxs}

    def save(self, p):
        pickle.dump(self, open(p, "w"))

    @staticmethod
    def load(p):
        if os.path.isfile(p):
            return pickle.load(open(p))
        else:
            return None


def tokenize(s, preserve_patterns=None, extrasubs=True):
    if not isinstance(s, unicode):
        s = s.decode("utf-8")
    s = unidecode.unidecode(s)
    repldic = None
    if preserve_patterns is not None:
        repldic = {}
        def _tokenize_preserve_repl(x):
            id = max(repldic.keys() + [-1]) + 1
            repl = "replreplrepl{}".format(id)
            assert(repl not in s)
            assert(id not in repldic)
            repldic[id] = x.group(0)
            return repl
        for preserve_pattern in preserve_patterns:
            s = re.sub(preserve_pattern, _tokenize_preserve_repl, s)
    if extrasubs:
        s = re.sub("[-_\{\}/]", " ", s)
    s = s.lower()
    tokens = nltk.word_tokenize(s)
    if repldic is not None:
        repldic = {"replreplrepl{}".format(k): v for k, v in repldic.items()}
        tokens = [repldic[token] if token in repldic else token for token in tokens]
    s = re.sub("`", "'", s)
    return tokens


def dataset(*x):
    if q.issequence(x):
        assert(len(x) == 1)
        x = x[0]
    # if not issequence(x):
    #     x = (x,)
    return tensor_dataset(*x)


def datacat(datasets, mode=1):
    if mode == 0:
        return torch.utils.data.dataset.ConcatDataset(datasets)
    elif mode == 1:
        return MultiDatasets(datasets)
    else:
        raise q.SumTingWongException("mode {} not recognized".format(mode))


class MultiDatasets(Dataset):
    def __init__(self, datasets):
        """ datasets to index together, result will be concatenated in one list """
        for xe in datasets:
            assert(len(xe) == len(datasets[0]))
        super(MultiDatasets, self).__init__()
        self.datasets = datasets

    def __getitem__(self, item):
        ret = tuple()
        for dataset in self.datasets:
            ret_a = dataset[item]
            if not isinstance(ret_a, tuple):
                ret_a = (ret_a,)
            ret += ret_a
        return ret

    def __len__(self):
        return len(self.datasets[0])


def tensor_dataset(*x):
    tensors = []
    for xe in x:
        if isinstance(xe, np.ndarray):
            xe = torch.tensor(xe)
        tensors.append(xe)
    for xe in tensors:
        assert(xe.size(0) == tensors[0].size(0))
    ret = torch.utils.data.dataset.TensorDataset(*tensors)
    return ret


def dataload(*tensors, **kw):
    if len(tensors) == 1 and isinstance(tensors[0], Dataset):
        tensordataset = tensors[0]
    else:
        tensordataset = tensor_dataset(*tensors)
    dataloader = torch.utils.data.DataLoader(tensordataset, **kw)
    return dataloader


def datasplit(npmats, splits=(80, 20), random=True):
    splits = np.round(len(npmats[0]) * np.cumsum(splits) / sum(splits)).astype("int32")

    whatsplit = np.zeros((len(npmats[0]),), dtype="int64")
    for i in range(1, len(splits)):
        a, b = splits[i-1], splits[i]
        whatsplit[a:b] = i

    if random is not False and random is not None:
        if isinstance(random, int):
            np.random.seed(random)
            random = True

        if random is True:
            np.random.shuffle(whatsplit)

    ret = []
    for i in range(0, len(splits)):
        splitmats = [npmat[whatsplit == i] for npmat in npmats]
        ret.append(splitmats)
    return ret


class ticktock(object):
    """ timer-printer thingy """
    def __init__(self, prefix="-", verbose=True):
        self.prefix = prefix
        self.verbose = verbose
        self.state = None
        self.perc = None
        self.prevperc = None
        self._tick()

    def tick(self, state=None):
        if self.verbose and state is not None:
            print("%s: %s" % (self.prefix, state))
        self._tick()

    def _tick(self):
        self.ticktime = dt.now()

    def _tock(self):
        return (dt.now() - self.ticktime).total_seconds()

    def progress(self, x, of, action="", live=False):
        if self.verbose:
            self.perc = int(round(100. * x / of))
            if self.perc != self.prevperc:
                if action != "":
                    action = " " + action + " -"
                topr = "%s:%s %d" % (self.prefix, action, self.perc) + "%"
                if live:
                    self._live(topr)
                else:
                    print(topr)
                self.prevperc = self.perc

    def tock(self, action=None, prefix=None):
        duration = self._tock()
        if self.verbose:
            prefix = prefix if prefix is not None else self.prefix
            action = action if action is not None else self.state
            print("%s: %s in %s" % (prefix, action, self._getdurationstr(duration)))
        return self

    def msg(self, action=None, prefix=None):
        if self.verbose:
            prefix = prefix if prefix is not None else self.prefix
            action = action if action is not None else self.state
            print("%s: %s" % (prefix, action))
        return self

    def _getdurationstr(self, duration):
        if duration >= 60:
            duration = int(round(duration))
            seconds = duration % 60
            minutes = (duration // 60) % 60
            hours = (duration // 3600) % 24
            days = duration // (3600*24)
            acc = ""
            if seconds > 0:
                acc = ("%d second" % seconds) + ("s" if seconds > 1 else "")
            if minutes > 0:
                acc = ("%d minute" % minutes) + ("s" if minutes > 1 else "") + (", " + acc if len(acc) > 0 else "")
            if hours > 0:
                acc = ("%d hour" % hours) + ("s" if hours > 1 else "") + (", " + acc if len(acc) > 0 else "")
            if days > 0:
                acc = ("%d day" % days) + ("s" if days > 1 else "") + (", " + acc if len(acc) > 0 else "")
            return acc
        else:
            return ("%.3f second" % duration) + ("s" if duration > 1 else "")

    def _live(self, x, right=None):
        if right:
            try:
                #ttyw = int(os.popen("stty size", "r").read().split()[1])
                raise Exception("qsdf")
            except Exception:
                ttyw = None
            if ttyw is not None:
                sys.stdout.write(x)
                sys.stdout.write(right.rjust(ttyw - len(x) - 2) + "\r")
            else:
                sys.stdout.write(x + "\t" + right + "\r")
        else:
            sys.stdout.write(x + "\r")
        sys.stdout.flush()

    def live(self, x):
        if self.verbose:
            self._live(self.prefix + ": " + x, "T: %s" % self._getdurationstr(self._tock()))

    def stoplive(self):
        if self.verbose:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


def argparsify(f, test=None):
    args, _, _, defaults = inspect.getargspec(f)
    assert(len(args) == len(defaults))
    parser = argparse.ArgumentParser()
    i = 0
    for arg in args:
        argtype = type(defaults[i])
        if argtype == bool:     # convert to action
            if defaults[i] == False:
                action="store_true"
            else:
                action="store_false"
            parser.add_argument("-%s" % arg, "--%s" % arg, action=action, default=defaults[i])
        else:
            parser.add_argument("-%s"%arg, "--%s"%arg, type=type(defaults[i]))
        i += 1
    if test is not None:
        par = parser.parse_args([test])
    else:
        par = parser.parse_args()
    kwargs = {}
    for arg in args:
        if getattr(par, arg) is not None:
            kwargs[arg] = getattr(par, arg)
    return kwargs


def argprun(f, sigint_shell=True, **kwargs):   # command line overrides kwargs
    """ use this to enable command-line access to kwargs of function (useful for main run methods) """
    def handler(sig, frame):
        # find the frame right under the argprun
        print("custom handler called")
        original_frame = frame
        current_frame = original_frame
        previous_frame = None
        stop = False
        while not stop and current_frame.f_back is not None:
            previous_frame = current_frame
            current_frame = current_frame.f_back
            if "_FRAME_LEVEL" in current_frame.f_locals \
                and current_frame.f_locals["_FRAME_LEVEL"] == "ARGPRUN":
                stop = True
        if stop:    # argprun frame found
            __toexposelocals = previous_frame.f_locals     # f-level frame locals
            class L(object):
                pass
            l = L()
            for k, v in __toexposelocals.items():
                setattr(l, k, v)
            stopprompt = False
            while not stopprompt:
                whattodo = raw_input("(s)hell, (k)ill\n>>")
                if whattodo == "s":
                    embed()
                elif whattodo == "k":
                    "Killing"
                    sys.exit()
                else:
                    stopprompt = True

    if sigint_shell:
        _FRAME_LEVEL="ARGPRUN"
        prevhandler = signal.signal(signal.SIGINT, handler)
    try:
        f_args = argparsify(f)
        for k, v in kwargs.items():
            if k not in f_args:
                f_args[k] = v
        f(**f_args)

    except KeyboardInterrupt:
        print("Interrupted by Keyboard")


def inp():
    return raw_input("Press ENTER to continue:\n>>> ")


def issequence(x):
    return isinstance(x, collections.Sequence) and not isinstance(x, basestring)


def iscollection(x):
    return issequence(x) or isinstance(x, set)


def isnumber(x):
    return isinstance(x, float) or isinstance(x, int)


def isstring(x):
    return isinstance(x, basestring)


def iscuda(x):
    if isinstance(x, torch.nn.Module):
        params = list(x.parameters())
        return params[0].is_cuda
    else:
        raise q.SumTingWongException("unsupported type")


def iscallable(x):
    return hasattr(x, "__call__")


def isfunction(x):
    return iscallable(x)


def getnumargs(f):
    return len(inspect.getargspec(f).args)


def getkw(kw, name, default=None, nodefault=False, remove=True):
    """ convenience function for getting certain kwargs out of function """
    if name in kw:
        ret = kw[name]
        if remove:
            del kw[name]
    else:
        if nodefault:
            raise Exception("kwarg {} must be specified (no default)".format(name))
        ret = default
    return ret



import torch
# pytorch utils


class var(object):
    """ shortcut for creating Variables and transferring cuda settings (cuda transfer limited to only gpu/cpu)
        Usage: q.var(torch.randn(5,4)).cuda(x).v --> will put result on gpu if x was on gpu"""
    all_cuda = False

    def __init__(self, x, requires_grad=False, volatile=False):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        self.v = torch.autograd.Variable(x, requires_grad=requires_grad, volatile=volatile)

    def cuda(self, crit=None):
        if isinstance(crit, int) and not isinstance(crit, bool):
            self.v = self.v.cuda(crit)
        elif crit is False:
            if self.v.is_cuda:
                self.v = self.v.cpu()
        elif crit is True:
            if not self.v.is_cuda:
                self.v = self.v.cuda()
        elif hasattr(crit, "is_cuda"):
            self.cuda(crit=crit.is_cuda)
        elif crit is None:
            self.cuda(crit=var.all_cuda)
        return self


class val(object):
    """ Shortcut for a value that should be saved on the nn.Module (persistent).
        Currently implemented as a nn.Parameter that does not require grad --> use q.params_of()
        Usage: q.val(torch.randn(5,4)).v"""
    def __init__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        self.v = torch.nn.Parameter(x, requires_grad=False)


class hyperparam(object):
    def __init__(self, initval):
        super(hyperparam, self).__init__()
        self._initval = initval
        self._v = initval

    def reset(self):
        self._v = self._initval

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v = value


def v(x):
    print("WARNING: don't use q.v()! ")
    if isinstance(x, hyperparam):
        return x._v
    elif isinstance(x, (var, val)):
        return x.v
    elif isinstance(x, torch.autograd.Variable):
        return x.data
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return x


def params_of(m):
    """ gets parameters of given nn.Module, filtering out params that don't require grad (q.val().v)"""
    params = m.parameters()
    params = filter(lambda x: x.requires_grad == True, params)
    return params


# QELOS TAGGING MECHANISM
def add_tag(x, tag):
    assert(isinstance(x, torch.autograd.Variable) and isstring(tag))
    add_qelos_key(x, "tags", set())
    x._qelos["tags"].add(tag)


def remove_tag(x, tag):
    assert(isinstance(x, torch.autograd.Variable) and isstring(tag))
    if hasattr(x, "_qelos") and "tags" in x._qelos:
        x._qelos["tags"].remove(tag)


def get_tags(x):
    assert(isinstance(x, torch.autograd.Variable))
    if hasattr(x, "_qelos") and "tags" in x._qelos:
        return x._qelos["tags"]


def filter_by_tag(xs, tag):
    assert(isstring(tag))
    for x in xs:
        if hasattr(x, "_qelos") and "tags" in x._qelos and tag in x._qelos["tags"]:
            yield x


def add_qelos_key(x, k, v=None):
    assert(isinstance(x, torch.autograd.Variable) and isstring(k))
    if not hasattr(x, "_qelos"):
        x._qelos = {}
    if k not in x._qelos:
        x._qelos[k] = v


def has_qelos_key(x, k):
    assert (isinstance(x, torch.autograd.Variable) and isstring(k))
    return hasattr(x, "_qelos") and k in x._qelos


def get_qelos_key(x, k):
    if has_qelos_key(x, k):
        return x._qelos[k]


def remove_qelos_key(x, k):
    if has_qelos_key(x, k):
        del x._qelos[k]


def paramgroups_of(m):
    params = params_of(m)
    default_group = {"params": []}
    paramgroups = []
    for param in params:
        g = None
        if has_qelos_key(param, "lr"):
            g = {"params": [param], "lr": get_qelos_key(param, "lr")}
        if has_qelos_key(param, "l2"):
            g = {"params": [param], "weight_decay": get_qelos_key(param, "l2")}
        if g is None:
            default_group["params"].append(param)
        else:
            paramgroups.append(g)
    paramgroups.append(default_group)
    return paramgroups


# SOME TRAINING SETTINGS
# - saved as qelos tags
# - must use q.paramgroups_of(...) to get qelos-tagged groups to torch.optim.Optimizer
def gradmult(xs, frac=1.):  # supports hyperparam as frac
    def hookf(_grad):
        return _grad * v(frac)
    if isinstance(xs, torch.autograd.Variable):
        xs = [xs]
    for xt in xs:
        remover = xt.register_hook(hookf)
        add_qelos_key(xt, "gradmult_removers", set())
        xt._qelos["gradmult_removers"].add(remover)


def remove_gradmult(xs):
    if isinstance(xs, torch.autograd.Variable):
        xs = [xs]
    for xt in xs:
        if hasattr(xt, "_qelos") and "gradmult_removers" in xt._qelos:
            for rmvr in xt._qelos["gradmult_removers"]:
                rmvr()
            del xt._qelos["gradmult_removers"]


def set_lr(x, lr):
    if isinstance(x, torch.nn.Module):
        for p in params_of(x):
            set_lr(p, lr)
    else:
        add_qelos_key(x, "lr", None)
        x._qelos["lr"] = lr


def remove_lr(x):
    if isinstance(x, torch.nn.Module):
        for p in params_of(x):
            remove_lr(p)
    else:
        remove_qelos_key(x, "lr")


def set_l2(x, l2):
    if isinstance(x, torch.nn.Module):
        for p in params_of(x):
            set_l2(p, l2)
    else:
        add_qelos_key(x, "l2", None)
        x._qelos["l2"] = l2


def remove_l2(x):
    if isinstance(x, torch.nn.Module):
        for p in params_of(x):
            remove_l2(p)
    else:
        remove_qelos_key(x, "l2")


# SEQUENCE PACKING AND UNPACKING
def seq_pack(x, mask):  # mask: (batsize, seqlen)
    """ given N-dim sequence "x" (N>=2), and 2D mask (batsize, seqlen)
        returns packed sequence (sorted) and indexes to un-sort (also used by seq_unpack) """
    x = x.float()
    mask = mask.float()
    # 1. get lengths
    lens = torch.sum(mask.float(), 1)
    # 2. sort by length
    assert(lens.dim() == 1)
    _, sortidxs = torch.sort(lens, descending=True)
    unsorter = torch.zeros(sortidxs.size()).to(sortidxs.device).long()
    # print ("test unsorter")
    # print (unsorter)
    unsorter.scatter_(0, sortidxs,
                           torch.arange(0, len(unsorter), dtype=torch.int64, device=sortidxs.device))
    # 3. pack
    sortedseq = torch.index_select(x, 0, sortidxs)
    sortedmsk = torch.index_select(mask, 0, sortidxs)
    sortedlens = sortedmsk.long().sum(1)
    sortedlens = list(sortedlens.cpu().detach().numpy())
    packedseq = torch.nn.utils.rnn.pack_padded_sequence(sortedseq, sortedlens, batch_first=True)
    return packedseq, unsorter


def seq_unpack(x, order, padding_value=0):
    """ given packed sequence "x" and the un-sorter "order",
        returns padded sequence (un-sorted by "order") and a binary 2D mask (batsize, seqlen),
            where padded sequence is padded with "padding_value" """
    unpacked, lens = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=padding_value)
    mask = torch.zeros(len(lens), max(lens), dtype=torch.int64, device=unpacked.device)
    for i, l in enumerate(lens):
        mask[i, :l] = 1
    out = torch.index_select(unpacked, 0, order)        # same as: unpacked[order]
    outmask = torch.index_select(mask, 0, order)        # same as: mask[order]
    return out, outmask



