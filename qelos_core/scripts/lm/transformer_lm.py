import torch
import qelos_core as q
import os
import random
from copy import deepcopy


def load_data(p="../../../datasets/wikitext2/",
              batsize=100, eval_batsize=10, seqlen=35):

    class Dictionary(object):
        def __init__(self):
            self.word2idx = {}
            self.idx2word = []

        def add_word(self, word):
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
            return self.word2idx[word]

        def __len__(self):
            return len(self.idx2word)

    class Corpus(object):
        def __init__(self, path):
            self.dictionary = Dictionary()
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))

        def tokenize(self, path):
            """Tokenizes a text file."""
            assert os.path.exists(path)
            # Add words to the dictionary
            with open(path, 'r', encoding="utf8") as f:
                tokens = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r', encoding="utf8") as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1

            return ids

    corpus = Corpus(p)

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data

    D = corpus.dictionary.word2idx
    train_data = LMLoader(corpus.train, seqlen, batsize=batsize)
    # valid_data = batchify(corpus.valid, eval_batsize)
    # valid_data = LMLoader_Test(valid_data, seqlen)
    valid_data = LMLoader_Test(corpus.valid, seqlen, batsize=batsize)
    # test_data = batchify(corpus.test, eval_batsize)
    # test_data = LMLoader_Test(test_data, seqlen)
    test_data = None
    return train_data, valid_data, test_data, D


# class LMLoader_Test(object):
#     """ data loader for LM data """
#     def __init__(self, data, seqlen):
#         super(LMLoader_Test, self).__init__()
#         self.data = data
#         self.seqlen = seqlen
#
#     def __iter__(self):
#         return _LMLoaderIter_Test(self)
#
#     def __len__(self):
#         return self.data.size(0)-1
#
#
# class _LMLoaderIter_Test(object):
#     def __init__(self, lmloader):
#         super(_LMLoaderIter_Test, self).__init__()
#         self.lml = lmloader
#         # self.i = 1
#         self.i = self.lml.seqlen+1
#
#     def __iter__(self):
#         return self
#
#     def __len__(self):
#         return len(self.lml)
#
#     def __next__(self):
#         if self.i < len(self.lml.data):
#             batch = self.lml.data[self.i-1]
#             batch_g = self.lml.data[self.i]
#             self.i += 1
#             return batch, batch_g
#         else:
#             self.i = 0
#             raise StopIteration()


class LMLoader(object):
    """ data loader for LM data """
    def __init__(self, data, seqlen, batsize):
        super(LMLoader, self).__init__()
        self.data = data
        self.seqlen = seqlen
        self.batsize = batsize

    def __iter__(self):
        return _LMLoaderIter(self)

    def __len__(self):
        return self.data.size(0) // (self.seqlen * self.batsize)


class _LMLoaderIter(object):
    def __init__(self, lmloader):
        super(_LMLoaderIter, self).__init__()
        self.lml = lmloader
        self.i = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.lml)

    def __next__(self):
        if self.i >= len(self.lml):
            raise StopIteration()
        self.i += 1
        out = []
        for k in range(self.lml.batsize):
            start = random.randint(0, self.lml.data.size(0) - self.lml.seqlen)
            out.append(self.lml.data[start: start+self.lml.seqlen])
        out = torch.stack(out, 0)
        gold = out[:, 1:]
        out = out[:, :-1]
        return out, gold


class LMLoader_Test(object):
    """ data loader for LM data """
    def __init__(self, data, seqlen, batsize):
        super(LMLoader_Test, self).__init__()
        self.data = data        # (totallen,)
        self.seqlen = seqlen
        self.batsize = batsize
        self.seglen = data.size(0) // batsize
        self.starts = [i*self.seglen for i in range(batsize)]
        d = [data[i: i+self.seglen] for i in self.starts]
        self._data = torch.stack(d, 0)

    def __iter__(self):
        return _LMLoaderIter_Test(self)

    def __len__(self):
        return self.data.size(0) // (self.batsize)


# class _LMLoaderIter_Test(object):
#     def __init__(self, lmloader):
#         super(_LMLoaderIter_Test, self).__init__()
#         self.lml = lmloader
#         self.i = 0
#
#     def __iter__(self):
#         return self
#
#     def __len__(self):
#         return len(self.lml)
#
#     def __next__(self):
#         if self.i + self.lml.seqlen >= self.lml.seglen:
#             raise StopIteration()
#         out = self.lml._data[:, self.i:self.i + self.lml.seqlen]
#         gold = out[:, -1].unsqueeze(1)
#         out = out[:, :-1]
#         self.i += 1
#         return out, gold


class _LMLoaderIter_Test(object):
    def __init__(self, lmloader):
        super(_LMLoaderIter_Test, self).__init__()
        self.lml = lmloader
        self.i = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.lml)

    def __next__(self):
        if self.i + 2 >= self.lml.seglen:
            raise StopIteration()
        out = self.lml._data[:, self.i:self.i + 1]
        gold = self.lml._data[:, self.i+1:self.i+2]
        self.i += 1
        return out, gold


# region rnn lm model -- don't need this for transformers
class LMModel(torch.nn.Module, q.AutoHooker):
    """
    A language model must implement autohooker such that:
     - all states are reset when going to a new epoch
     - all states are transferred in between batches
        - reload and backup states manually in forward !!!

    Must be used in conjunction with LMLoader
    """
    def get_hooks(self, emitter):
        return {
            emitter.START_EPOCH:    self.reset_backup_states,
        }

    def reset_backup_states(self, _, **kw):
        """ reset backed up states of model """
        raise NotImplemented("use subclass")


class RNNLayer_LM(LMModel):
    encodertype = q.LSTMEncoder

    def __init__(self, *dims:int, worddic:dict=None, bias:bool=True, dropout:float=0., **kw):
        super(RNNLayer_LM, self).__init__(**kw)
        self.dims = dims
        self.D = worddic
        self.states = None
        # make layers
        self.emb = q.WordEmb(dims[0], worddic=self.D)
        self.out = q.WordLinout(dims[-1], worddic=self.D)
        self.rnn = self.encodertype(*dims, bidir=False, bias=bias, dropout_in=dropout)
        self.rnn.ret_all_states = True
        self.dropout = torch.nn.Dropout(p=dropout)

    def reset_backup_states(self, _, **kw):
        self.states = None

    def forward(self, x):
        emb, xmask = self.emb(x)
        # do actual forward
        states_0 = ((None, None) if self.encodertype == q.LSTMEncoder else (None,)) \
            if self.states is None else self.states
        out, all_states = self.rnn._forward(emb, mask=xmask, states_0=states_0, ret_states=True)
        # backup states
        all_states = [[all_state_e.detach() for all_state_e in all_state] for all_state in all_states]
        self.states = list(zip(*all_states))

        # output
        out = self.dropout(out)
        out = self.out(out)
        return out
# endregion


# region transformer language model
class TransformerLM(torch.nn.Module):
    def __init__(self, dim=512, worddic=None, numlayers=3, numheads=8, activation=torch.nn.ReLU,
                 embedding_dropout=0., attention_dropout=0., residual_dropout=0.,
                 word_dropout=0., relpos=True, tie_wordvecs=True, maxlen=512):
        super(TransformerLM, self).__init__()
        self.wordemb = q.WordEmb(dim, worddic=worddic, word_dropout=word_dropout)
        self.transformer = q.TransformerDecoder(dim=dim, numlayers=numlayers, numheads=numheads, activation=activation,
                                                embedding_dropout=embedding_dropout, attention_dropout=attention_dropout,
                                                residual_dropout=residual_dropout, relpos=relpos, noctx=True, maxlen=maxlen)
        self.wordout = q.WordLinout(dim, worddic=worddic)
        if tie_wordvecs:
            self.wordout.lin.weight = self.wordemb.embedding.weight

    def forward(self, x):   # (batsize, seqlen) wordids
        xemb, _ = self.wordemb(x)
        enc = self.transformer(xemb)
        out = self.wordout(enc)
        return out


class TransformerLMCell(torch.nn.Module):
    def __init__(self, core:TransformerLM, horizon:int=100):
        super(TransformerLMCell, self).__init__()
        self.core = q.deep_copy(core, share_params=True)
        self.core.transformer = q.TransformerDecoderCell(self.core.transformer, horizon)
        self.horizon = horizon

    def forward(self, x):   # (batsize, 1) wordids
        out = self.core(x)
        return out

# class TransformerLMCell(torch.nn.Module):
#     def __init__(self, core:TransformerLM, horizon:int=100):
#         super(TransformerLMCell, self).__init__()
#         self.core = core
#         self.horizon = horizon
#
#     def forward(self, x):
#         """
#         :param x:    (batsize, seqlen)
#         :return:     (batsize, vocsize)
#         """
#         out = self.core(x)
#         ret = out[:, -1].unsqueeze(1)
#         return ret
# endregion


def run(lr=0.001,
        edropout=0.2,
        wdropout=0.1,
        rdropout=0.3,
        adropout=0.05,
        numlayers=3,
        numheads=8,
        relpos=True,
        tie_wordvecs=True,
        gradnorm=5.,
        epochs=100,
        dim=128,
        seqlen=50,
        batsize=64,
        eval_batsize=64,
        cuda=False,
        gpu=0,
        test=True,
        ):
    tt = q.ticktock("script")
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda", gpu)
    tt.tick("loading data")
    train_batches, valid_batches, test_batches, D = \
        load_data(batsize=batsize, eval_batsize=eval_batsize, seqlen=seqlen)
    tt.tock("data loaded")
    print("{} batches in train".format(len(train_batches)))

    tt.tick("creating model")

    m = TransformerLM(dim=dim, worddic=D, numlayers=numlayers, numheads=numheads,
                      activation=torch.nn.ReLU, embedding_dropout=edropout,attention_dropout=adropout,
                      word_dropout=wdropout, residual_dropout=rdropout, relpos=relpos,
                      tie_wordvecs=tie_wordvecs, maxlen=2*seqlen)
    valid_m = TransformerLMCell(m, seqlen)
    q.copy_params(m, valid_m.core)

    if test:
        for i, batch in enumerate(valid_batches):
            y = valid_m(batch[0])
            if i > 5:
                break
        for i, batch in enumerate(valid_batches):
            pass
        print(i, batsize, seqlen, valid_batches.data.size(0))
        print(y.size())
    # return

    loss = q.SeqKLLoss(time_average=True, size_average=True, mode="logits")
    test_loss = loss
    ppl_loss = q.SeqPPLLoss(time_average=True, size_average=True, mode="logits")
    # test_loss = q.KLLoss(size_average=True, mode="logits")
    # ppl_loss = q.PPLLoss(size_average=True, mode="logits")

    # optim = torch.optim.SGD(q.params_of(m), lr=lr)
    optim = torch.optim.Adam(q.params_of(m), lr=lr)
    gradclip = q.ClipGradNorm(gradnorm)
    lrp = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=1/2, patience=0, verbose=True)

    trainer = q.trainer(m).on(train_batches).loss(loss).optimizer(optim).device(device).hook(m).hook(gradclip)
    tester = q.tester(valid_m).on(valid_batches).loss(test_loss, ppl_loss).device(device).hook(m)

    tt.tock("created model")
    tt.tick("training")
    q.train(trainer, tester).hook(lrp, tester.losses[1])\
        .run(epochs=epochs)
    tt.tock("trained")

    tt.tick("testing")
    finaltester = q.tester(valid_m).on(test_batches).loss(test_loss, ppl_loss).device(device).hook(m)
    finaltester.run()
    tt.tock("tested")


if __name__ == '__main__':
    q.argprun(run)