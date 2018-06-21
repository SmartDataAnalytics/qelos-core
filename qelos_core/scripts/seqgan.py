import torch
import qelos_core as q
import numpy as np

EPS = 1e-8


class Decoder(torch.nn.Module):
    """ self-sampling decoder
        basically free-running decoder, returns sampled sequence and probabilities
    """
    def __init__(self, D, embdim, zdim, startsym, *innerdim, **kw):
        super(Decoder, self).__init__()
        self.emb = q.WordEmb(embdim, worddic=D)
        innerdim = (embdim+zdim,) + innerdim
        self.layers = torch.nn.ModuleList(modules=[
            q.LSTMCell(innerdim[i-1], innerdim[i]) for i in range(1, len(innerdim))
        ])
        self.linout = q.WordLinout(innerdim[-1], worddic=D)
        self.sm = torch.nn.Softmax(-1)
        self.maxtime = q.getkw(kw, "maxtime", 100)
        self.startid = D[startsym]
        self.sm_sample = True

    def forward(self, z, maxtime=None):
        maxtime = self.maxtime if maxtime is None else maxtime
        p_ts = []
        s_ts = []
        s_tm1 = torch.ones(z.size(0)).long() * self.startid
        for t in range(maxtime):
            s_tm1_emb, _mask = self.emb(s_tm1)
            #z = torch.zeros_like(z)     # TODO: REMOVE (debugging)
            y_t = torch.cat([s_tm1_emb, z], 1)
            for layer in self.layers:
                y_t = layer(y_t)
            p_t = self.linout(y_t)
            p_t = self.sm(p_t)      # (batsize, vocsize)
            if self.sm_sample:
                p_t_dist = torch.distributions.Categorical(probs=p_t)
                s_t = p_t_dist.sample()
            else:
                _, s_t = p_t.max(1)
            p_ts.append(p_t.unsqueeze(1))
            s_ts.append(s_t.unsqueeze(1))
            s_tm1 = s_t
        p_ts = torch.cat(p_ts, 1)
        s_ts = torch.cat(s_ts, 1)
        return s_ts, p_ts


class Discriminator(torch.nn.Module):
    def __init__(self, D, embdim, *innerdim, **kw):
        super(Discriminator, self).__init__()
        self.emb = q.WordEmb(embdim, worddic=D)
        self.core = q.FastestLSTMEncoder(embdim, *innerdim)
        self.outlin1 = torch.nn.Linear(innerdim[-1], innerdim[-1])
        self.outlin1_sigm = torch.nn.Sigmoid()
        self.outlin2 = torch.nn.Linear(innerdim[-1], 1)
        self.outlin2_sigm = torch.nn.Sigmoid()

    def forward(self, x):   # (batsize, seqlen) int ids~vocsize
        x_emb, x_mask = self.emb(x)
        encs = self.core(x_emb)
        outs = self.outlin1_sigm(self.outlin1(encs))
        outs = self.outlin2_sigm(self.outlin2(outs)).squeeze(-1)
        # outs: (batsize, seqlen)
        return outs


class SeqGAN_Base(q.gan.GAN):
    def __init__(self, discriminator, decoder, gan_mode=None, rebasehalf=False, accumulate=True, critic=None):
        super(SeqGAN_Base, self).__init__(discriminator, decoder, gan_mode=gan_mode)
        self.rebasehalf = rebasehalf
        self.accumulate = accumulate

    def forward_disc_train(self, x, z):
        real_score = self.discriminator(*x)[:, -1]
        fake_sym, fake_probs = self.generator(*z)
        fake = fake_sym
        fake_score = self.discriminator(fake)[:, -1]
        real_loss = - torch.log(real_score.clamp(EPS, np.infty))
        fake_loss = - torch.log((1 - fake_score).clamp(EPS, np.infty))
        loss = real_loss + fake_loss
        return loss, torch.zeros_like(loss)

    def forward_gen_train(self, z):
        fake_sym, fake_probs = self.generator(z)
        fake_scores = self.discriminator(fake_sym).detach()[:, -1]

        if self.rebasehalf:
            fake_scores = (fake_scores - 0.5) * 2

        logprobs = torch.gather(fake_probs, 2, fake_sym.unsqueeze(2)).squeeze(2)
        logprobs = - torch.log(logprobs.clamp(EPS, np.infty))

        reward = fake_scores
        loss = logprobs * reward.unsqueeze(1)
        return loss, reward


# TODO: something wrong happens here regarding contribs
class SeqGAN_DCL(q.gan.GAN):
    def __init__(self, discriminator, decoder, gan_mode=None, rebasehalf=False, accumulate=True, critic=None):
        super(SeqGAN_DCL, self).__init__(discriminator, decoder, gan_mode=gan_mode)
        self.rebasehalf = rebasehalf
        self.accumulate = accumulate
        self.critic = critic        # TODO: remove

    def forward_disc_train(self, x, z):
        real_score = self.discriminator(*x)

        fake_sym, fake_probs = self.generator(*z)
        fake = fake_sym
        fake_score = self.discriminator(fake)

        #fake_score += - torch.min(fake_score) + 0.05  # TODO: for debugging --> REMOVE

        contribs, endcontribs = self.get_contribs(fake_score)

        real_loss = - torch.log(real_score.clamp(EPS, np.infty))
        fake_loss = - torch.log((1 - fake_score).clamp(EPS, np.infty))
        loss = real_loss + fake_loss
        loss = loss * contribs
        loss = loss.sum(1)
        return loss, endcontribs.float()

    def get_contribs(self, scores):     # TODO: _contribs must always sum up to one???
        topcontrib = 1.
        scores = scores.detach()
        certainties = 1 + scores * torch.log2(scores.clamp(EPS, np.infty)) \
                      + (1 - scores) * torch.log2((1 - scores).clamp(EPS, np.infty))
        contribs = torch.cumsum(certainties, 1)
        _contribs = contribs.clamp(0, topcontrib)
        maxcontribs = _contribs.max(1)[0].unsqueeze(1)
        _contribs = torch.cat([torch.zeros(_contribs.size(0), 1, device=_contribs.device), _contribs], 1)
        _contribs = _contribs[:, 1:] - _contribs[:, :-1]

        basecontribs = torch.ones_like(scores) * (topcontrib - maxcontribs) / scores.size(1)
        _contribs = _contribs + basecontribs
        contrib_sums = _contribs.sum(1)

        # where do contribs stop?
        contribs_ = torch.cat([torch.zeros(contribs.size(0), 1, device=_contribs.device), _contribs], 1)
        endcontrib = torch.cumsum(contribs_, 1) >= (1 - 1e-3)
        endcontrib = endcontrib[:, 1:] - endcontrib[:, :-1]
        _endcontrib = torch.nonzero(endcontrib)[:, 1] + 1
        if _endcontrib.size(0) != scores.size(0):
            assert(_endcontrib.size(0) == scores.size(0))
        return _contribs, _endcontrib

    def forward_gen_train(self, z):
        fake_sym, fake_probs = self.generator(z)
        fake_scores = self.discriminator(fake_sym).detach()

        #fake_scores += - torch.max(fake_scores) + 0.85        # TODO: for debugging --> REMOVE

        contribs, endcontrib = self.get_contribs(fake_scores)

        # gather probabilities of sampled sequence
        logprobs = torch.gather(fake_probs, 2, fake_sym.unsqueeze(2)).squeeze(2)
        logprobs = - torch.log(logprobs.clamp(EPS, np.infty))

        # compute scores to be used in update
        _fake_scores = fake_scores
        if self.rebasehalf:
            _fake_scores = (fake_scores - 0.5) * 2
        _fake_scores = _fake_scores * contribs      # use contribs to ignore irrelevant future
                                                    # TODO: is this correct? ignoring well-learned past?

        if self.accumulate:     # accumulate without horizon
            seqlen = _fake_scores.size(1)
            advantage = torch.cumsum(_fake_scores[:, seqlen - torch.arange(seqlen, dtype=torch.int64) - 1], 1) \
                [:, seqlen - torch.arange(seqlen, dtype=torch.int64) - 1]
        else:                           # no accumulation
            advantage = _fake_scores

        # compute loss
        loss = logprobs * advantage
        loss = loss.sum(1)
        return loss, endcontrib.float()


def test_local():
    # test decoder
    words = "<MASK> <START> a b c d e".split()
    wD = dict(zip(words, range(len(words))))
    decoder = Decoder(wD, 50, 50, "<START>", 40)
    sample_z = torch.randn(4, 50)
    s_ts, p_ts = decoder(sample_z)
    # test discriminator
    discr = Discriminator(wD, 50, 40)
    scores = discr(s_ts)

    # test seqgan dcl
    decoder.maxtime = 10
    seqgan = SeqGAN_DCL(discr, decoder, accumulate=True)
    q.batch_reset(seqgan)
    seqgan._gan_mode = SeqGAN_DCL.DISC_TRAIN
    ret = seqgan(torch.randint(1, 7, (4, 10), dtype=torch.int64), sample_z)
    q.batch_reset(seqgan)
    seqgan._gan_mode = SeqGAN_DCL.GEN_TRAIN
    ret = seqgan(sample_z)


def gen_toy_data(N, seqlen=10, mode="oneletter"):      # abcdefg <-- letters allowed
    import random
    vocab = "<START> a b c d e f g".split()
    vocab = dict(zip(vocab, range(len(vocab))))
    ret = []
    for i in range(N):
        if mode == "oneletter":     # just one letter
            ret.append("a"*seqlen)
        elif mode == "twosameletters":
            letter = random.choice(["a", "b"])
            ret.append(letter*seqlen)
        elif mode == "threesameletters":
            letter = random.choice(["a", "b", "c"])
            ret.append(letter*seqlen)
        elif mode == "foursameletters":
            letter = random.choice(["a", "b", "c", "d"])
            ret.append(letter*seqlen)
        elif mode == "sameletter":    # different letters repeated over whole length
            letter = random.choice(list(set(vocab.keys()) - set(["<START>"])))
            ret.append(letter*seqlen)
        elif mode == "oneinterleave":
            letter = "ab"
            app = (letter * seqlen)[:seqlen]
            ret.append(app)
        elif mode == "oneinterleaveboth":
            letter = random.choice(["ab", "ba"])
            app = (letter * seqlen)[:seqlen]
            ret.append(app)
        elif mode == "twointerleave":
            letter = random.choice(["ac", "dc"])
            app = (letter * seqlen)[:seqlen]
            ret.append(app)
        elif mode == "twointerleaveboth":
            letter = random.choice(["ac", "ca", "dc", "cd"])
            app = (letter * seqlen)[:seqlen]
            ret.append(app)
        elif mode == "fixedstartend":
            middle = []
            for i in range(seqlen-2):
                middle.append(random.choice(list(set(vocab.keys()) - set(["<START>"]))))
            app = "a" + "".join(middle) + "a"
            ret.append(app)
        elif mode == "copymiddlefixed":
            ptrn = random.choice(["c{}cc{}ccc", "cc{}c{}ccc", "c{}cccc{}c"])
            letter = random.choice(["a", "b", "d", "e", "f", "g"])
            app = ptrn.format(letter, letter)
            ret.append(app)
        elif mode == "kebab":
            app = random.choice(["cebab", "dagab", "dageg"])
            app += "f"*(seqlen-len(app))
            ret.append(app)
        elif mode == "repeathalf":
            prefix = []
            for i in range((seqlen+1)//2):
                prefix.append(random.choice(list(set(vocab.keys()) - set(["<START>"]))))
            prefix = "".join(prefix)
            ret.append((prefix + prefix)[:seqlen])
    return ret, vocab


def run_words(lr=0.001,
              seqlen=8,
              batsize=50,
              epochs=1000,
              embdim=64,
              innerdim=128,
              z_dim=64,
              usebase=True,
              noaccumulate=False,
              ):
    # get some words
    N = 1000
    glove = q.PretrainedWordEmb(50, vocabsize=N+2)
    words = glove.D.keys()[2:]
    datasm = q.StringMatrix()
    datasm.tokenize = lambda x: list(x)
    for word in words:
        datasm.add(word)
    datasm.finalize()
    datamat = datasm.matrix[:, :seqlen]
    # replace <mask> with <end>
    datamat = datamat + (datamat == datasm.D["<MASK>"]) * (datasm.D["<END>"] - datasm.D["<MASK>"])


    real_data = q.dataset(datamat)
    gen_data_d = q.gan.gauss_dataset(z_dim, len(real_data))
    disc_data = q.datacat([real_data, gen_data_d], 1)

    gen_data = q.gan.gauss_dataset(z_dim)

    disc_data = q.dataload(disc_data, batch_size=batsize, shuffle=True)
    gen_data = q.dataload(gen_data, batch_size=batsize, shuffle=True)

    discriminator = Discriminator(datasm.D, embdim, innerdim)
    generator = Decoder(datasm.D, embdim, z_dim, "<START>", innerdim, maxtime=seqlen)

    SeqGAN = SeqGAN_Base if usebase else SeqGAN_DCL

    disc_model = SeqGAN(discriminator, generator, gan_mode=q.gan.GAN.DISC_TRAIN, accumulate=not noaccumulate)
    gen_model = SeqGAN(discriminator, generator, gan_mode=q.gan.GAN.GEN_TRAIN, accumulate=not noaccumulate)

    disc_optim = torch.optim.Adam(q.params_of(discriminator), lr=lr)
    gen_optim = torch.optim.Adam(q.params_of(generator), lr=lr)

    disc_trainer = q.trainer(disc_model).on(disc_data).optimizer(disc_optim).loss(q.no_losses(2))
    gen_trainer = q.trainer(gen_model).on(gen_data).optimizer(gen_optim).loss(q.no_losses(2))

    gan_trainer = q.gan.GANTrainer(disc_trainer, gen_trainer)

    gan_trainer.run(epochs, disciters=5, geniters=1, burnin=500)

    # print some predictions:
    with torch.no_grad():
        rvocab = {v: k for k, v in datasm.D.items()}
        q.batch_reset(generator)
        eval_z = torch.randn(50, z_dim)
        eval_y, _ = generator(eval_z)
        for i in range(len(eval_y)):
            prow = "".join([rvocab[mij] for mij in eval_y[i].numpy()])
            print(prow)

    print("done")


def run_toy(lr=0.001,
            seqlen=8,
            batsize=10,
            epochs=1000,
            embdim=32,
            innerdim=64,
            z_dim=32,
            noaccumulate=False,
            usebase=False,
            ):
    # generate some toy data
    N = 1000
    data, vocab = gen_toy_data(N, seqlen=seqlen, mode="copymiddlefixed")
    datasm = q.StringMatrix()
    datasm.set_dictionary(vocab)
    datasm.tokenize = lambda x: list(x)
    for data_e in data:
        datasm.add(data_e)
    datasm.finalize()

    real_data = q.dataset(datasm.matrix)
    gen_data_d = q.gan.gauss_dataset(z_dim, len(real_data))
    disc_data = q.datacat([real_data, gen_data_d], 1)

    gen_data = q.gan.gauss_dataset(z_dim)

    disc_data = q.dataload(disc_data, batch_size=batsize, shuffle=True)
    gen_data = q.dataload(gen_data, batch_size=batsize, shuffle=True)

    discriminator = Discriminator(datasm.D, embdim, innerdim)
    generator = Decoder(datasm.D, embdim, z_dim, "<START>", innerdim, maxtime=seqlen)

    SeqGAN = SeqGAN_Base if usebase else SeqGAN_DCL

    disc_model = SeqGAN(discriminator, generator, gan_mode=q.gan.GAN.DISC_TRAIN, accumulate=not noaccumulate)
    gen_model = SeqGAN(discriminator, generator, gan_mode=q.gan.GAN.GEN_TRAIN, accumulate=not noaccumulate)

    disc_optim = torch.optim.Adam(q.params_of(discriminator), lr=lr)
    gen_optim = torch.optim.Adam(q.params_of(generator), lr=lr)

    disc_trainer = q.trainer(disc_model).on(disc_data).optimizer(disc_optim).loss(q.no_losses(2))
    gen_trainer = q.trainer(gen_model).on(gen_data).optimizer(gen_optim).loss(q.no_losses(2))

    gan_trainer = q.gan.GANTrainer(disc_trainer, gen_trainer)

    gan_trainer.run(epochs, disciters=5, geniters=1, burnin=500)

    # print some predictions:
    with torch.no_grad():
        rvocab = {v: k for k, v in vocab.items()}
        q.batch_reset(generator)
        eval_z = torch.randn(50, z_dim)
        eval_y, _ = generator(eval_z)
        for i in range(len(eval_y)):
            prow = "".join([rvocab[mij] for mij in eval_y[i].numpy()])
            print(prow)

    print("done")


if __name__ == "__main__":
    q.argprun(run_words)