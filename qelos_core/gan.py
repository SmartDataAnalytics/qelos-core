import qelos_core as q
import torch
from torch.utils.data.dataset import Dataset


class SampleDataset(Dataset):
    def __init__(self, sample_function, size=1e6):
        """ sample function must accept batsize arg and return tensor with (batsize, ...) dims
            every call to __getitem__ samples a new random value
        """
        super(SampleDataset, self).__init__()
        self.sf = sample_function
        self.length = size

    def __getitem__(self, item):
        if isinstance(item, int):
            ret = self.sf(1).squeeze(0)
        else:
            ret = self.sf(len(item))
        return ret

    def __len__(self):
        return self.length


def gauss_dataset(dim, size=1e6):
    """
    Creates a dataloader of randomly sampled gaussian noise
    The returned dataloader produces batsize batches of dim-sized vectors
    """
    def samplef(bsize):
        return torch.randn(bsize, dim)

    ret = SampleDataset(samplef, size=size)
    return ret


class GAN(torch.nn.Module):
    DISC_TRAIN = 1
    GEN_TRAIN = 2

    def __init__(self, discriminator, generator, gan_mode=None):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self._gan_mode = gan_mode

    def forward_disc_train(self, x, z):
        real_score = self.discriminator(*x)
        fake = self.generator(*z)
        fake = fake.detach()
        fake_score = self.discriminator(fake)
        loss = self.disc_loss(real_score, fake_score)
        return loss

    def disc_loss(self, real_score, fake_score, *args, **kw):
        return - torch.log(real_score) - torch.log(1 - fake_score)

    def forward_gen_train(self, *z):
        fake = self.generator(*z)
        fake_score = self.discriminator(fake)
        loss = self.gen_loss(fake_score)
        return loss

    def gen_loss(self, fake_score, *args, **kw):
        return - torch.log(fake_score)

    def forward(self, *x):
        if self._gan_mode == self.DISC_TRAIN:
            return self.forward_disc_train(*x)
        elif self._gan_mode == self.GEN_TRAIN:
            return self.forward_gen_train(*x)
        else:
            return self.generate(1)


class GANTrainer(q.LoopRunner, q.EventEmitter):
    START = 0
    END = 1
    START_EPOCH = 2
    END_EPOCH = 3
    START_TRAIN = 4
    END_TRAIN = 5
    START_VALID = 6
    END_VALID = 7
    START_DISC = 8
    END_DISC = 9
    START_GEN = 10
    END_GEN = 11

    def __init__(self, disc_trainer, gen_trainer, validator=None):
        """
        Creates a GAN trainer given a gen_trainer and disc_trainer.
        both trainers already contain the model, optimizer and losses and implement updating and batching
        """
        super(GANTrainer, self).__init__()
        self.disc_trainer = disc_trainer
        self.gen_trainer = gen_trainer
        self.validator = validator
        self.stop_training = False

    def runloop(self, iters, disciters=1, geniters=1, validinter=1, burnin=10):
        tt = q.ticktock("gan runner")
        self.do_callbacks(self.START)
        current_iter = 0
        disc_batch_iter = self.disc_trainer.inf_batches(with_info=False)
        gen_batch_iter = self.gen_trainer.inf_batches(with_info=False)
        while self.stop_training is not True:
            tt.tick()
            self.do_callbacks(self.START_EPOCH)
            self.do_callbacks(self.START_TRAIN)
            self.do_callbacks(self.START_DISC)

            _disciters = burnin if current_iter == 0 else disciters

            for disc_iter in range(_disciters):  # discriminator iterations
                batch = disc_batch_iter.next()
                self.disc_trainer.do_batch(batch)
                ttmsg = "iter {}/{} - disc: {}/{} - {}".format(current_iter, iters,
                                                               disc_iter, _disciters,
                                                               self.disc_trainer.losses.pp())
                tt.live(ttmsg)
            tt.stoplive()
            self.do_callbacks(self.END_DISC)
            self.do_callbacks(self.START_GEN)
            for gen_iter in range(geniters):  # generator iterations
                batch = gen_batch_iter.next()
                self.gen_trainer.do_batch(batch)
                ttmsg = "iter {}/{} - gen: {}/{} - {}".format(current_iter, iters,
                                                              gen_iter, geniters,
                                                              self.gen_trainer.losses.pp())
                tt.live(ttmsg)
            tt.stoplive()
            self.do_callbacks(self.END_GEN)
            ttmsg = "iter {}/{} - disc: {} - gen: {}".format(current_iter, iters,
                                                             self.disc_trainer.losses.pp(),
                                                             self.gen_trainer.losses.pp())
            self.disc_trainer.losses.push_and_reset()
            self.gen_trainer.losses.push_and_reset()
            self.do_callbacks(self.END_TRAIN)

            if self.validator is not None and current_iter % validinter == 0:
                self.do_callbacks(self.START_VALID)
                if isinstance(self.validator, q.tester):
                    self.validator.do_epoch()
                    ttmsg += " -- {}".format(self.validator.losses.pp())
                else:
                    toprint = self.validator()
                    ttmsg += " -- {}".format(toprint)
                self.do_callbacks(self.END_VALID)
            self.do_callbacks(self.END_EPOCH)
            tt.tock(ttmsg)
            current_iter += 1
            self.stop_training = current_iter >= iters
        self.do_callbacks(self.END)

    def run(self, iters, disciters=1, geniters=1, validinter=1, burnin=10):
        self.stop_training = False
        self.gen_trainer.pre_run()
        self.disc_trainer.pre_run()
        if isinstance(self.validator, q.tester):
            self.validator.pre_run()
        self.runloop(iters, disciters=disciters, geniters=geniters, validinter=validinter, burnin=burnin)
        self.gen_trainer.post_run()
        self.disc_trainer.post_run()
        if isinstance(self.validator, q.tester):
            self.validator.post_run()