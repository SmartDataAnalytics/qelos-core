import torch
import qelos_core as q


class Decoder(torch.nn.Module):
    """ self-sampling decoder """
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

    def forward(self, x):
        pass


def run(lr=0.001):
    # test decoder
    words = "<MASK> <START> a b c d e".split()
    wD = dict(zip(words, range(len(words))))
    decoder = Decoder(wD, 50, 50, "<START>", 40)
    sample_z = torch.randn(4, 50)
    s_ts, p_ts = decoder(sample_z)


if __name__ == "__main__":
    q.argprun(run)