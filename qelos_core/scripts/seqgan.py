import torch
import qelos_core as q


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
        self.maxtime = 100
        self.startid = D[startsym]

    def forward(self, z, maxtime=None):
        maxtime = self.maxtime if maxtime is None else maxtime
        p_ts = []
        s_ts = []
        s_tm1 = torch.ones(z.size(0)).long() * self.startid
        for t in range(maxtime):
            s_tm1_emb, _mask = self.emb(s_tm1)
            y_t = torch.cat([s_tm1_emb, z], 1)
            for layer in self.layers:
                y_t = layer(y_t)
            p_t = self.linout(y_t)
            p_t = self.sm(p_t)      # (batsize, vocsize)
            p_t_dist = torch.distributions.Categorical(probs=p_t)
            s_t = p_t_dist.sample()
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


# TODO: use discriminator certainty contribs in discriminator training
# TODO: add actor critic
class SeqGAN_DCL(q.gan.GAN):
    def __init__(self, discriminator, decoder, gan_mode=None, rebasehalf=True, accumulate=True, critic=None):
        super(SeqGAN_DCL, self).__init__(discriminator, decoder, gan_mode=gan_mode)
        self.rebasehalf = rebasehalf
        self.accumulate = accumulate
        self.critic = critic        # TODO

    def forward_disc_train(self, x, z):
        real_score = self.discriminator(*x)

        fake_sym, fake_probs = self.generator(*z)
        fake = fake_sym
        fake_score = self.discriminator(fake)

        real_loss = - torch.log(real_score)
        fake_loss = - torch.log(1 - fake_score)
        loss = (real_loss - fake_loss).sum(1)
        # TODO
        return loss

    def get_contribs(self, scores):
        certainties = 1 + scores * torch.log2(scores) + (1 - scores) * torch.log2(1 - scores)
        certainties = certainties.detach()
        contribs = torch.cumsum(certainties, 1)
        _contribs = contribs.clamp(0, 1)
        _contribs = torch.cat([torch.zeros(_contribs.size(0), 1), _contribs], 1)
        _contribs = _contribs[:, 1:] - _contribs[:, :-1]
        contrib_sums = _contribs.sum(1)

        # where do contribs stop?
        contribs_ = torch.cat([contribs, 10 * torch.ones(contribs.size(0), 1)], 1)
        endcontrib = contribs_ > 1
        endcontrib = endcontrib[:, 1:] - endcontrib[:, :-1]
        endcontrib = torch.nonzero(endcontrib)[:, 1] + 1
        assert(endcontrib.size(0) == scores.size(0))
        return _contribs, endcontrib

    def forward_gen_train(self, z):
        fake_sym, fake_probs = self.generator(z)
        fake_scores = self.discriminator(fake_sym).detach()

        fake_scores += - torch.max(fake_scores) + 0.85        # TODO: for debugging --> REMOVE

        contribs, endcontrib = self.get_contribs(fake_scores)

        # gather probabilities of sampled sequence
        logprobs = torch.gather(fake_probs, 2, fake_sym.unsqueeze(2)).squeeze(2)
        logprobs = - torch.log(logprobs)

        # compute scores to be used in update
        _fake_scores = fake_scores
        if self.rebasehalf:
            _fake_scores = (fake_scores - 0.5) * 2
        _fake_scores = _fake_scores * contribs      # use contribs to ignore irrelevant future

        if self.accumulate:     # accumulate without horizon
            seqlen = _fake_scores.size(1)
            advantage = torch.cumsum(_fake_scores[:, seqlen - torch.arange(seqlen, dtype=torch.int64) - 1], 1) \
                [:, seqlen - torch.arange(seqlen, dtype=torch.int64) - 1]
        else:                           # no accumulation
            advantage = _fake_scores

        # compute loss
        loss = logprobs * advantage
        loss = loss.sum(1)
        return loss


def run(lr=0.001):
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
    seqgan = SeqGAN_DCL(discr, decoder, advantage=None)
    q.batch_reset(seqgan)
    seqgan._gan_mode = SeqGAN_DCL.DISC_TRAIN
    ret = seqgan(torch.randint(1, 7, (4, 10), dtype=torch.int64), sample_z)
    q.batch_reset(seqgan)
    seqgan._gan_mode = SeqGAN_DCL.GEN_TRAIN
    ret = seqgan(sample_z)


    print("done")


if __name__ == "__main__":
    q.argprun(run)