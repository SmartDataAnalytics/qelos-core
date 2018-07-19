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


def log_prob_seq_standard_gauss(x, mask=None):
    """
    Computes log prob of given points under standard gauss N(0, I)
    :param x:   (batsize, seqlen, gauss_dim) batch of samples
    """
    ret = - 0.5 * ((x**2).sum(2) + x.size(2) * np.log(2*np.pi))
    if mask is not None:
        ret = ret * mask.float()
    ret = ret.sum(1)
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


class SimpleIAT(torch.nn.Module):
    """ Lower-triangular matrix for computing mus and sigmas"""
    def __init__(self, dim, **kw):
        super(SimpleIAT, self).__init__(**kw)
        self.W_mu = torch.nn.Parameter(torch.empty(dim, dim))
        self.W_sigma = torch.nn.Parameter(torch.empty(dim, dim))
        self.b_sigma = torch.nn.Parameter(torch.ones(1, dim))
        self.mu_0 = torch.nn.Parameter(torch.randn(1)*0.1)
        self.sigma_0 = torch.nn.Parameter(torch.randn(1)*0.1)
        self.W_mask = torch.triu(torch.ones(dim, dim), 1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.W_mu)
        torch.nn.init.xavier_normal_(self.W_sigma)

    def forward(self, x, x_dens=None, x_mask=None):
        """
        :param x:           (batsize, dim) or (batsize, seqlen, dim)
        :param x_dens:      (batsize,)
        :param x_mask:      (batsize, seqlen), if x was (batsize, seqlen, dim)
        :return:
        """
        assert(x_mask is None or x.dim() == 3)
        mus = torch.matmul(x, self.W_mu * self.W_mask)
        b_sigma = self.b_sigma if x.dim() == 2 else self.b_sigma.unsqueeze(1)
        sigmas = torch.matmul(x, self.W_sigma * self.W_mask) + b_sigma
        sigmas = torch.nn.functional.sigmoid(sigmas)
        y = x * sigmas + mus * (1 - sigmas)
        d_dens = - torch.log(sigmas).sum(-1)       # (batsize,) or (batsize, seqlen)
        if x_mask is not None:
            d_dens = d_dens * x_mask.float()
        if x.dim() == 3:
            d_dens = d_dens.sum(1)      # (batsize,)
        y_dens = x_dens + d_dens
        return y, y_dens


class SimpleSeqIAT(torch.nn.Module):
    """ RNN-based IAT for sequences """
    def __init__(self, core, coredim, outdim, **kw):
        """
        :param core:        RNN
        :param dim:
        :param kw:
        """
        super(SimpleSeqIAT, self).__init__(**kw)
        self.W_mu = torch.nn.Parameter(torch.empty(coredim, outdim))
        self.W_sigma = torch.nn.Parameter(torch.empty(coredim, outdim))
        self.b_sigma = torch.nn.Parameter(torch.ones(1, 1, outdim))
        self.mu_0 = torch.nn.Parameter(torch.randn(1, 1, outdim) * 0.1)
        self.sigma_0 = torch.nn.Parameter(torch.randn(1, 1, outdim) * 0.1)
        self.core = core
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.W_mu)
        torch.nn.init.xavier_normal_(self.W_sigma)

    def forward(self, x, x_dens=None, x_mask=None):
        """
        :param x:           (batsize, seqlen, indim)
        :param x_dens:      (batsize,)
        :param x_mask:      (batsize, seqlen)
        :return:
        """
        y = self.core(x)        # (batsize, seqlen, indim), y_t only depends on x_{0:t-1}
        mus = torch.matmul(y, self.W_mu)            # (batsize, seqlen, outdim)
        sigmas = torch.matmul(y, self.W_sigma) + self.b_sigma
        sigmas = torch.nn.functional.sigmoid(sigmas)
        z = x[:, 1:] * sigmas[:, :-1] + mus[:, :-1] * (1 - sigmas[:, :-1])
        sigma_0 = torch.nn.functional.sigmoid(self.sigma_0)
        x_0 = x[:, 0:1] * sigma_0 + self.mu_0
        z = torch.cat([x_0, z], 1)
        d_dens = - (torch.log(sigmas[:, :-1]).sum(-1) + torch.log(sigma_0).sum(-1))   # (batsize, seqlen)
        if x_mask is not None:
            d_dens = d_dens * x_mask.float()
        d_dens = d_dens.sum(1)      # (batsize,)
        y_dens = x_dens + d_dens
        return z, y_dens


class Posterior(torch.nn.Module):
    def __init__(self, core, outdim, z_dim):
        super(Posterior, self).__init__()
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
        return z, log_posterior


class SeqPosterior(Posterior):
    def forward(self, x, x_mask=None):
        out = self.core(x)   # (batsize, seqlen, encdim)
        mu = self.mu_net(out)   # (batsize, seqlen, zdim)
        sigma = self.sigma_net(out) # (batsize, seqlen, zdim)
        eps = torch.randn_like(sigma)
        mu, sigma, eps = mu[:, 1:], sigma[:, 1:], eps[:, 1:]
        x_mask = x_mask[:, 1:] if x_mask is not None else None
        z = mu + eps * sigma
        log_posterior = log_prob_seq_gauss(z, mu, sigma, x_mask)
        return z, log_posterior


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
        self._debug = True

    def forward(self, x, x_mask=None):
        z, log_posterior = self.encoder(x, x_mask=x_mask)
        _x_mask = x_mask[:, 1:] if x_mask is not None else None
        if z.dim() == 2:
            log_prior = log_prob_standard_gauss(z)
            x_hat = self.decoder(x[:, :-1], z=z)
        elif z.dim() == 3:
            log_prior = log_prob_seq_standard_gauss(z, mask=_x_mask)
            x_hat = self.decoder([x[:, :-1], z])
        else:
            raise q.SumTingWongException("z must be 2D or 3D, got {}D".format(z.dim()))
        log_likelihood = self.likelihood(x_hat, x[:, 1:], x_mask=_x_mask)
        kl_div = log_posterior - log_prior
        elbo = log_likelihood - kl_div
        rets = -elbo, kl_div, -log_likelihood
        if self._debug:
            z_grad = torch.autograd.grad(log_likelihood.sum(), z, retain_graph=True)
            z_grad = z_grad[0]**2
            if z.dim() == 3:
                z_grad = z_grad.sum(2)
            z_grad = z_grad.sum(1) ** 0.5
            rets = rets + (z_grad,)
        return rets


def run_seqvae_toy(lr=0.001,
                   embdim=64,
                   encdim=100,
                   zdim=64,
                   batsize=50,
                   epochs=100,
                   ):

    # test
    vocsize = 100
    seqlen = 12
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
            out, states = self.core(embs, mask, ret_states=True)
            top_state = states[-1][0][:, 0]
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

        def forward(self, xs):
            x, z = xs
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

    x = torch.randint(0, vocsize, (batsize, seqlen), dtype=torch.int64)
    ys = vae(x)

    optim = torch.optim.Adam(q.params_of(vae), lr=lr)

    x = torch.randint(0, vocsize, (batsize * 100, seqlen), dtype=torch.int64)
    dataloader = q.dataload(x, batch_size=batsize, shuffle=True)

    trainer = q.trainer(vae).on(dataloader).optimizer(optim).loss(4).epochs(epochs)
    trainer.run()

    print("done \n\n")
    # DONE: add IAF
    # TODO: add normal VAE for seqs
    # TODO: experiments


def run_normal_seqvae_toy(lr=0.001,
                   embdim=64,
                   encdim=100,
                   zdim=64,
                   batsize=50,
                   epochs=100,
                   ):

    # test
    vocsize = 100
    seqlen = 12
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
            out, states = self.core(embs, mask, ret_states=True)
            top_state = states[-1][0][:, 0]
            # top_state = top_state.unsqueeze(1).repeat(1, out.size(1), 1)
            return top_state      # (batsize, encdim)

    encoder_net = EncoderNet(encoder_emb, encoder_lstm)
    encoder = Posterior(encoder_net, encdim, zdim)
    # endregion

    # region decoder
    decoder_emb = q.WordEmb(embdim, worddic=wD)
    decoder_lstm = q.LSTMCell(embdim+zdim, encdim)
    decoder_outlin = q.WordLinout(encdim, worddic=wD)

    class DecoderCell(torch.nn.Module):
        def __init__(self, emb, core, out, **kw):
            super(DecoderCell, self).__init__()
            self.emb, self.core, self.out = emb, core, out

        def forward(self, xs, z=None):
            embs, mask = self.emb(xs)
            core_inp = torch.cat([embs, z], 1)
            core_out = self.core(core_inp)
            out = self.out(core_out)
            return out

    decoder_cell = DecoderCell(decoder_emb, decoder_lstm, decoder_outlin)
    decoder = q.TFDecoder(decoder_cell)
    # endregion

    likelihood = Likelihood()

    vae = SeqVAE(encoder, decoder, likelihood)

    x = torch.randint(0, vocsize, (batsize, seqlen), dtype=torch.int64)
    ys = vae(x)

    optim = torch.optim.Adam(q.params_of(vae), lr=lr)

    x = torch.randint(0, vocsize, (batsize * 100, seqlen), dtype=torch.int64)
    dataloader = q.dataload(x, batch_size=batsize, shuffle=True)

    trainer = q.trainer(vae).on(dataloader).optimizer(optim).loss(4).epochs(epochs)
    trainer.run()

    print("done \n\n")


if __name__ == "__main__":
    # q.argprun(run_seqvae_toy)
    q.argprun(run_normal_seqvae_toy)