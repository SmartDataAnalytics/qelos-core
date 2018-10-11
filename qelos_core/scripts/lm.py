import torch
import qelos_core as q
import os


def load_data(p="../../datasets/wikitext2/",
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

    train_data = batchify(corpus.train, batsize)
    valid_data = batchify(corpus.valid, eval_batsize)
    test_data = batchify(corpus.test, eval_batsize)
    D = corpus.dictionary.word2idx
    train_data = LMLoader(train_data, seqlen)
    valid_data = LMLoader(valid_data, seqlen)
    test_data = LMLoader(test_data, seqlen)
    return train_data, valid_data, test_data, D


class LMLoader(object):
    """ data loader for LM data """
    def __init__(self, data, seqlen):
        super(LMLoader, self).__init__()
        self.data = data
        self.seqlen = seqlen

    def __iter__(self):
        return _LMLoaderIter(self)

    def __len__(self):
        return 1 + ( (len(self.data)-1) // self.seqlen)


class _LMLoaderIter(object):
    def __init__(self, lmloader):
        super(_LMLoaderIter, self).__init__()
        self.lml = lmloader
        self.i = 0

    def __iter__(self):
        return self

    def __len__(self):
        return 1 + ((len(self.lml.data)-1) // self.lml.seqlen)

    def __next__(self):
        if self.i < len(self.lml.data)-1:
            seqlen = min(self.lml.seqlen, len(self.lml.data) - self.i - 1)
            batch = self.lml.data[self.i: self.i + seqlen]
            batch_g = self.lml.data[self.i+1: self.i+1 + seqlen]
            self.i += seqlen
            return batch.transpose(1, 0), batch_g.transpose(1, 0)
        else:
            self.i = 0
            raise StopIteration()


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
        self.states = zip(*all_states)
        # output
        out = self.out(out)
        return out


def run(lr=0.001,
        dropout=0.2,
        gradnorm=0.25,
        epochs=25,
        embdim = 200,
        encdim = 200,
        numlayers = 3,
        seqlen=35,
        batsize=20,
        eval_batsize=10,
        cuda=False,
        gpu=0,
        test=False
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
    dims = [embdim] + ([encdim] * numlayers)
    m = RNNLayer_LM(*dims, worddic=D, dropout=dropout)

    if test:
        for i, batch in enumerate(train_batches):
            y = m(batch[0])
            if i > 5:
                break
        print(y.size())

    loss = q.SeqKLLoss(time_average=True, size_average=True, mode="logits")
    ppl_loss = q.SeqPPL_Loss(time_average=True, size_average=True, mode="logits")

    optim = torch.optim.Adam(q.params_of(m), lr=lr)
    gradclip = q.ClipGradNorm(gradnorm)

    trainer = q.trainer(m).on(train_batches).loss(loss).optimizer(optim).device(device).hook(m).hook(gradclip)
    tester = q.tester(m).on(valid_batches).loss(loss, ppl_loss).device(device).hook(m)

    tt.tock("created model")
    tt.tick("training")
    q.train(trainer, tester).run(epochs=epochs)
    tt.tock("trained")


if __name__ == '__main__':
    q.argprun(run)