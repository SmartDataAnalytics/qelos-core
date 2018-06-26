import qelos_core as q
import torch
from torch.utils.data.dataset import Dataset
import torchvision
import numpy as np


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


class InceptionForEval(torch.nn.Module):
    def __init__(self, normalize_input=True, resize_input=True):
        super(InceptionForEval, self).__init__()
        self.inception = torchvision.models.inception_v3(pretrained=True)
        self.inception.eval()       # set to eval mode
        self.layers = torch.nn.Sequential(
            self.inception.Conv2d_1a_3x3,
            self.inception.Conv2d_2a_3x3,
            self.inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(3, stride=2),
            self.inception.Conv2d_3b_1x1,
            self.inception.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(3, stride=2),
            self.inception.Mixed_5b,
            self.inception.Mixed_5c,
            self.inception.Mixed_5d,
            self.inception.Mixed_6a,
            self.inception.Mixed_6b,
            self.inception.Mixed_6c,
            self.inception.Mixed_6d,
            self.inception.Mixed_6e,
            self.inception.Mixed_7a,
            self.inception.Mixed_7b,
            self.inception.Mixed_7c,
            torch.nn.AvgPool2d(8)
        )
        self.normalize_input = normalize_input
        self.resize_input = resize_input

    def forward(self, x):
        """ run the forward of inception layer, take prefinal activations as well as outputs """
        if self.resize_input:
            x = torch.nn.functional.upsample(x, size=(299, 299), mode='bilinear')
        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        prefinal = self.layers(x)
        prefinal = torch.nn.functional.dropout(prefinal, training=self.training)
        prefinal = prefinal.view(prefinal.size(0), -1)      # 2048
        outprobs = self.inception.fc(prefinal)
        return outprobs.detach(), prefinal.detach()


class InceptionMetric(object):
    def __init__(self, inception=None, device=torch.device("cpu")):
        super(InceptionMetric, self).__init__()
        self.inception = inception if inception is not None else InceptionForEval()
        self.device = device
        self.inception = self.inception.to(device)
        self.inception.eval()   # put adapted inception network officially in eval model


class FID(InceptionMetric):

    def get_activations(self, data):     # dataloader
        tocat = []
        for batch in data:
            batch = torch.tensor(batch, device=self.device)
            probs, activations = self.inception(batch)
            tocat.append(activations)
        allactivations = torch.cat(tocat, 0)
        return allactivations

    def get_activation_stats(self, data):
        activations = self.get_activations(data).to(torch.device("cpu")).detach().numpy()
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def get_distance(self, real_data, gen_data, eps=1e-6):
        mu1, sigma1 = self.get_activation_stats(gen_data)
        mu2, sigma2 = self.get_activation_stats(real_data)

        # from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = np.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = np.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)


class IS(InceptionMetric):

    def get_scores(self, data, splits=10):     # dataloader
        allprobs = []
        for batch in data:
            batch = torch.tensor(batch, device=self.device)
            print(batch.device, self.device)
            scores, activations = self.inception(batch)

            probs = torch.nn.functional.softmax(scores).detach()
            allprobs.append(probs)
        allprobs = torch.cat(allprobs, 0)
        allprobs = allprobs.detach().cpu().numpy()

        scores = []
        for i in range(splits):
            part = allprobs[(i * allprobs.shape[0] // splits):((i + 1) * allprobs.shape[0] // splits), :]
            part_means = np.expand_dims(np.mean(part, 0), 0)
            kl = part * (np.log(part) - np.log(part_means))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))

        return np.mean(scores), np.std(scores)



