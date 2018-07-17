import qelos_core as q
import torch
import numpy as np


def log_prob_standard_gauss(x):
    """
    Computes log prob of given points under standard gauss N(0, I)
    :param x:   (batsize, gauss_dim) batch of samples
    """
    ret = - 0.5 * ((x**2).sum(1) + x.size(1) * np.log(2*np.pi))
    return ret


def log_prob_gauss(x, mu, sigma):
    """
    Computes log prob of given points given diagonal gaussians N(mu, sigma)
    :param x:   (batsize, gauss_dim)
    :param mu:  (batsize, gauss_dim)
    :param sigma: (batsize, gauss_dim)
    """
    ret = - 0.5 * (torch.log(sigma).sum(1)
                   + ((x - mu)**2 * sigma**(-1)).sum(1)
                   + x.size(1) * np.log(2*np.pi))
    return ret


def log_prob_seq_gauss(x, mu, sigma, x_mask=None):
    """
    :param x:       (batsize, seqlen, zdim)
    :param mu:      (batsize, seqlen, zdim)
    :param sigma:   (batsize, seqlen, zdim)
    :param x_mask:  (batsize, seqlen)
    :return:
    """
    ret = - 0.5 * (torch.log(sigma).sum(2)
                   + ((x - mu)**2 * sigma**(-1)).sum(2)
                   + x.size(2) * np.log(2*np.pi))
    # (batsize, seqlen)
    if x_mask is not None:
        ret = ret * x_mask.float()

    ret = ret.sum(1)        # (batsize,)
    return ret


class Encoder(torch.nn.Module):
    def __init__(self, core, outdim, z_dim):
        super(Encoder, self).__init__()
        self.core = core
        self.mu_net = torch.nn.Sequential(torch.nn.Linear(outdim, z_dim))
        self.sigma_net = torch.nn.Sequential(torch.nn.Linear(outdim, z_dim),
                                             torch.nn.Softplus())

    def forward(self, x, x_mask=None):
        out = self.core(x)
        mu = self.mu_net(out)
        sigma = self.sigma_net(out)
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        log_posterior = log_prob_gauss(z, mu, sigma)
        return out, log_posterior


class SeqPosterior(Encoder):
    def forward(self, x, x_mask=None):
        out = self.core(x)   # (batsize, seqlen, encdim)
        mu = self.mu_net(out)   # (batsize, seqlen, zdim)
        sigma = self.sigma_net(out) # (batsize, seqlen, zdim)
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        log_posterior = log_prob_seq_gauss(z, mu, sigma, x_mask)
        return out, log_posterior


class Likelihood(torch.nn.Module):
    """ Sequence CE loss """
    def __init__(self):
        super(Likelihood, self).__init__()
        self.sm = torch.nn.Softmax(-1)

    def forward(self, x_hat, x, x_mask=None):
        """
        :param x_hat:   probabilities per time step (batsize, seqlen, outvocsize)
        :param x:       (batsize, seqlen) integer ids of true sequence
        :param x_mask:  (batsize, seqlen) mask (optional)
        :return:
        """
        x_hat = self.sm(x_hat)
        out = torch.gather(x_hat, 2, x.unsqueeze(2)).squeeze(2)     # (batsize, seqlen)
        out = torch.log(out)
        if x_mask is not None:
            out = out * x_mask.float()
        out = out.sum(1)
        return out


class SeqVAE(torch.nn.Module):
    def __init__(self, encoder, decoder, likelihood, **kw):
        """
        :param encoder:     produces sample z and its log_posterior
        :param decoder:     produces x_hat given x and z
        :param likelihood:  produces log_likelihood given x and x_hat
        :param kw:
        """
        super(SeqVAE, self).__init__(**kw)
        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood

    def forward(self, x, x_mask=None):
        z, log_posterior = self.encoder(x, x_mask=x_mask)
        log_prior = log_prob_standard_gauss(z)
        x_hat = self.decoder(x, z)
        log_likelihood = self.likelihood(x_hat, x, x_mask=x_mask)
        elbo = log_likelihood - log_posterior + log_prior
        return -elbo, log_posterior - log_prior


def run_seq(lr=0.001,
            embdim=100,
            encdim=100,
            zdim=50,
            ):

    # test
    batsize = 10
    vocsize = 100
    seqlen = 6
    x = torch.randint(0, vocsize, (batsize, seqlen), dtype=torch.int64)
    wD = dict((chr(xi), xi) for xi in range(vocsize))

    # region encoder
    encoder_emb = q.WordEmb(embdim, worddic=wD)
    encoder_lstm = q.FastestLSTMEncoder(embdim, encdim)

    class EncoderNet(torch.nn.Module):
        def __init__(self, emb, core):
            super(EncoderNet, self).__init__()
            self.emb, self.core = emb, core

        def forward(self, x):
            embs, mask = self.emb(x)
            out = self.core(embs, mask)
            return out      # (batsize, seqlen, encdim)

    encoder_net = EncoderNet(encoder_emb, encoder_lstm)
    encoder = SeqPosterior(encoder_net, encdim, zdim)
    # endregion

    # region decoder
    decoder_emb = q.WordEmb(embdim, worddic=wD)
    decoder_lstm = q.LSTMCell(embdim+zdim, encdim)
    decoder_outlin = q.WordLinout(encdim, worddic=wD)

    class DecoderCell(torch.nn.Module):
        def __init__(self, emb, core, out, **kw):
            super(DecoderCell, self).__init__()
            self.emb, self.core, self.out = emb, core, out

        def forward(self, x, z):
            embs, mask = self.emb(x)
            core_inp = torch.cat([embs, z], 1)
            core_out = self.core(core_inp)
            out = self.out(core_out)
            return out

    decoder_cell = DecoderCell(decoder_emb, decoder_lstm, decoder_outlin)
    decoder = q.TFDecoder(decoder_cell)
    # endregion

    likelihood = Likelihood()

    vae = SeqVAE(encoder, decoder, likelihood)

    ys = vae(x)


if __name__ == "__main__":
    q.argprun(run_seq)