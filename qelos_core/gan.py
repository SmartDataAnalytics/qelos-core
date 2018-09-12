import qelos_core as q
import torch
from torch.utils.data.dataset import Dataset
import torchvision
import numpy as np
from scipy import linalg
import json
import os
from PIL import Image


class SampleDataset(Dataset):
    def __init__(self, sample_function, size=1e6):
        """ sample function must accept batsize arg and return tensor with (batsize, ...) dims
            every call to __getitem__ samples a new random value
        """
        super(SampleDataset, self).__init__()
        self.sf = sample_function
        self.length = int(size)

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
        real_score = self.discriminator(x)
        fake = self.generator(z)
        fake = fake.detach()
        fake_score = self.discriminator(fake)
        loss = self.disc_loss(real_score, fake_score, x, fake)
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

    def disc_train(self):       # switches model for discriminator training
        self._gan_mode = self.DISC_TRAIN
        return self

    def gen_train(self):        # switches model for generator training
        self._gan_mode = self.GEN_TRAIN
        return self


class WGAN(GAN):
    def __init__(self, critic, gen, gan_mode=None, mode="LP", lamda=5):
        super(WGAN, self).__init__(critic, gen, gan_mode=gan_mode)
        self.mode = mode
        self.lamda = lamda

    def disc_loss(self, real_score, fake_score, real, fake, *args, **kw):
        core = - (real_score - fake_score)
        interp_alpha = torch.rand(real.size(0), 1, 1, 1, device=real_score.device)
        interp_points = interp_alpha * real + (1 - interp_alpha) * fake
        interp_points = interp_points.detach()
        interp_points.requires_grad = True
        interp_score = self.discriminator(interp_points)
        interp_grad, = torch.autograd.grad(interp_score, interp_points,
                                           grad_outputs=torch.ones_like(interp_score),
                                           create_graph=True)
        interp_grad_norm = interp_grad.view(interp_grad.size(0), -1).norm(p=2, dim=1)
        if self.mode == "LP":
            penalty = (interp_grad_norm - 1).clamp(0, np.infty) ** 2
        elif self.mode == "GP":
            penalty = (interp_grad_norm - 1) ** 2
        penalty = self.lamda * penalty
        loss = core + penalty
        return loss, core, penalty

    def gen_loss(self, fake_score, *args, **kw):
        return - fake_score


class WGAN_F(WGAN):
    def __init__(self, critic, gen, featurer, gan_mode=None, mode="LP", lamda=5,
                 pixel_penalty=False, **kw):
        super(WGAN_F, self).__init__(critic, gen, gan_mode=gan_mode, mode=mode, lamda=lamda, **kw)
        self.featurer = featurer
        self.pixel_penalty = pixel_penalty

    def forward_disc_train(self, x, z):
        _x = self.featurer(x)
        real_score = self.discriminator(_x)
        fake = self.generator(z)
        fake = fake.detach()
        _fake = self.featurer(fake)
        fake_score = self.discriminator(_fake)
        if self.pixel_penalty:      # implements pixel value based gradient penalty as usual
            _x, _fake = x, fake
        loss = self.disc_loss(real_score, fake_score, _x, _fake)
        return loss

    def disc_loss(self, real_score, fake_score, real, fake):
        core = - (real_score - fake_score)
        interp_alpha = torch.rand(real.size(0), 1, 1, 1, device=real_score.device)
        interp_points = interp_alpha * real + (1 - interp_alpha) * fake
        interp_points = interp_points.detach()
        interp_points.requires_grad = True
        if self.pixel_penalty:
            _interp_points = self.featurer(interp_points)
        else:
            _interp_points = interp_points
        interp_score = self.discriminator(_interp_points)
        interp_grad, = torch.autograd.grad(interp_score, interp_points,
                                           grad_outputs=torch.ones_like(interp_score),
                                           create_graph=True)
        interp_grad_norm = interp_grad.view(interp_grad.size(0), -1).norm(p=2, dim=1)
        if self.mode == "LP":
            penalty = (interp_grad_norm - 1).clamp(0, np.infty) ** 2
        elif self.mode == "GP":
            penalty = (interp_grad_norm - 1) ** 2
        penalty = self.lamda * penalty
        loss = core + penalty
        return loss, core, penalty

    def forward_gen_train(self, *z):
        fake = self.generator(*z)
        _fake = self.featurer(fake)
        fake_score = self.discriminator(_fake)
        loss = self.gen_loss(fake_score)
        return loss


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

    def __init__(self, disc_trainer, gen_trainer, validators=None, lr_decay=False):
        """
        Creates a GAN trainer given a gen_trainer and disc_trainer.
        both trainers already contain the model, optimizer and losses and implement updating and batching

        Takes a validator or a list of validators (with different validinters).
        """
        super(GANTrainer, self).__init__()
        self.disc_trainer = disc_trainer
        self.gen_trainer = gen_trainer
        if not q.issequence(validators) and validators is not None:
            validators = (validators,)
        self.validators = validators
        self.stop_training = False
        self.lr_decay = lr_decay

    def runloop(self, iters, disciters=1, geniters=1, burnin=10):
        tt = q.ticktock("gan runner")
        self.do_callbacks(self.START)
        current_iter = 0
        disc_batch_iter = self.disc_trainer.inf_batches(with_info=False)
        gen_batch_iter = self.gen_trainer.inf_batches(with_info=False)

        lr_decay_disc, lr_decay_gen = None, None
        if self.lr_decay:
            lr_decay_disc = torch.optim.lr_scheduler.LambdaLR(self.disc_trainer.optim,
                                                              lr_lambda=lambda ep: max(0, 1. - ep*1./iters))
            lr_decay_gen = torch.optim.lr_scheduler.LambdaLR(self.gen_trainer.optim,
                                                              lr_lambda=lambda ep: max(0, 1. - ep*1./iters))

        while self.stop_training is not True:
            tt.tick()
            self.do_callbacks(self.START_EPOCH)
            self.do_callbacks(self.START_TRAIN)
            self.do_callbacks(self.START_DISC)

            if lr_decay_disc is not None:
                lr_decay_disc.step()
            if lr_decay_gen is not None:
                lr_decay_gen.step()

            _disciters = burnin if current_iter == 0 else disciters

            for disc_iter in range(_disciters):  # discriminator iterations
                batch = next(disc_batch_iter)
                self.disc_trainer.do_batch(batch)
                ttmsg = "iter {}/{} - disc: {}/{} :: {}".format(current_iter, iters,
                                                               disc_iter+1, _disciters,
                                                               self.disc_trainer.losses.pp())
                tt.live(ttmsg)
            tt.stoplive()
            self.do_callbacks(self.END_DISC)
            self.do_callbacks(self.START_GEN)
            for gen_iter in range(geniters):  # generator iterations
                batch = next(gen_batch_iter)
                self.gen_trainer.do_batch(batch)
                ttmsg = "iter {}/{} - gen: {}/{} :: {}".format(current_iter, iters,
                                                              gen_iter+1, geniters,
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

            if self.validators is not None:
                for validator in self.validators:
                    if current_iter % validator.validinter == 0:
                        self.do_callbacks(self.START_VALID)
                        if isinstance(validator, q.tester):
                            validator.do_epoch()
                            ttmsg += " -- {}".format(validator.losses.pp())
                        else:
                            toprint = validator(iter=current_iter)
                            ttmsg += " -- {}".format(toprint)
                        self.do_callbacks(self.END_VALID)
            self.do_callbacks(self.END_EPOCH)
            tt.tock(ttmsg)
            current_iter += 1
            self.stop_training = current_iter >= iters
        self.do_callbacks(self.END)

    def run(self, iters, disciters=1, geniters=1, burnin=10):
        self.stop_training = False
        self.gen_trainer.pre_run()
        self.disc_trainer.pre_run()
        for validator in self.validators:
            if hasattr(validator, "pre_run"):
                validator.pre_run()
        self.runloop(iters, disciters=disciters, geniters=geniters, burnin=burnin)
        self.gen_trainer.post_run()
        self.disc_trainer.post_run()
        for validator in self.validators:
            if hasattr(validator, "post_run"):
                validator.post_run()


def make_img(arr, size=None):
    """
    :param img: numpy array containing a single image in RGB, values in range (-1, +1), DxHxW
    :param size: if specified, automatically determines a upscale/downscale factor and resizes image
    :return: image object that can be shown or saved
    """
    arr = (arr/0.5 + 0.5) * 255
    arr = arr.transpose(1, 2, 0)
    height, width, _ = arr.shape
    img = Image.fromarray(arr, "RGB")

    if size is not None:
        scale = size * 1. / max(height, width)
        newsize = (int(round(scale * width)), int(round(scale * height)))
        img = img.resize(newsize, Image.ANTIALIAS)

    return img


class Validator(object):
    pass    # TODO: validator for monitoring disc and gen scores on a dev set


class GeneratorValidator(object):
    """ Validator for generator. Runs generator on gendata and executes scorers on generated data """
    def __init__(self, generator, scorers, gendata, device=torch.device("cpu"), logger=None, validinter=1):
        """
        :param generator:   the generator
        :param scorers:     scorers (FID, IS, Imagesaver)
        :param gendata:     dataloader of data to feed to generator to generate images
        :param device:      device used only for batches (generator/scorers are not set to this device)
        """
        super(GeneratorValidator, self).__init__()
        self.history = {}
        self.generator = generator
        self.scorers = scorers
        self.gendata = gendata
        self.device = device
        self.tt = q.ticktock("validator")
        self._iter = 0
        self.logger = logger
        self.validinter = validinter

    def __call__(self, iter=None):
        iter = self._iter if iter is None else iter
        self.generator.eval()
        with torch.no_grad():
            # collect generated images
            generated = []
            self.tt.tick("running generator")
            for i, batch in enumerate(self.gendata):
                batch = (batch,) if not q.issequence(batch) else batch
                batch = [torch.tensor(batch_e).to(self.device) for batch_e in batch]
                _gen = self.generator(*batch).detach().cpu()
                _gen = _gen[0] if q.issequence(_gen) else _gen
                generated.append(_gen)
                self.tt.live("{}/{}".format(i, len(self.gendata)))
            batsize = max(map(len, generated))
            generated = torch.cat(generated, 0)
            self.tt.tock("generated data")

            gen_loaded = q.dataload(generated, batch_size=batsize, shuffle=False)
            rets = [iter]
            for scorer in self.scorers:
                ret = scorer(gen_loaded)
                if ret is not None:
                    rets.append(ret)
            if self.logger is not None:
                self.logger.liner_write("validator.txt", " ".join(map(str, rets)))
            self._iter += 1
        return " ".join(map(str, rets[1:]))

    def post_run(self):
        if self.logger is not None:
            self.logger.liner_close("validator")


class GenDataSaver(object):
    """ saves generated data as a single ndarray, overwrites previously saved data """
    def __init__(self, logger=None, p="saved", **kw):
        super(GenDataSaver, self).__init__(**kw)
        self.p = p; self.logger=logger

    def __call__(self, gendata):
        ret = []
        for batch in gendata:
            if not q.issequence(batch):
                batch = (batch,)
            ret.append(batch)
        ret = [[batch_e.cpu() for batch_e in batch] for batch in ret]
        ret = [torch.cat(ret_i, 0).numpy() for ret_i in zip(*ret)]
        tosave = dict(zip(map(str, range(len(ret))), ret))
        if self.logger is not None:
            np.savez(os.path.join(self.logger.p, self.p), **tosave)


class InceptionV2ForEval(torch.nn.Module):
    def __init__(self, resize_input=True, normalize_input=False):
        super(InceptionV2ForEval, self).__init__()
        assert(normalize_input == False)
        self.resize_input = resize_input
        self.inception = q.ganutil.inceptionresnetv2(num_classes=1000, pretrained="imagenet")
        self.inception.eval()

    def forward(self, x):
        """ input is RGB, channel first, range=(-1, 1)"""
        if self.resize_input:
            x = torch.nn.functional.upsample(x, size=(299, 299), mode='bilinear')
        # x = (x + 1.) * 127.5
        y, acts = self.inception(x, with_activations=True)
        return y.detach(), acts.detach()


class InceptionForEval(torch.nn.Module):
    def __init__(self, normalize_input=False, resize_input=True):
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
        """ run the forward of inception layer, take prefinal activations as well as outputs
            x: in range (-1, 1)
        """
        if self.resize_input:
            x = torch.nn.functional.upsample(x, size=(299, 299), mode='bilinear')
        if self.normalize_input:
            x = x.clone()
            x[:, 0] = (x[:, 0] * 0.229 + 0.485 - 0.5) / 0.5
            x[:, 1] = (x[:, 1] * 0.224 + 0.456 - 0.5) / 0.5
            x[:, 2] = (x[:, 2] * 0.225 + 0.406 - 0.5) / 0.5
        prefinal = self.layers(x)
        # prefinal = torch.nn.functional.dropout(prefinal, training=self.training)
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

    def get_inception_outs(self, data):     # dataloader
        tt = q.ticktock("inception")
        tt.tick("running data through network")
        probses = []
        activationses = []
        for i, batch in enumerate(data):
            batch = (batch,) if not q.issequence(batch) else batch
            batch = [torch.tensor(batch_e).to(self.device) for batch_e in batch]
            probs, activations = self.inception(*batch)
            probs = torch.nn.functional.softmax(probs)
            probses.append(probs.detach())
            activationses.append(activations.detach())
            tt.live("{}/{}".format(i, len(data)))
        tt.stoplive()
        tt.tock("done")
        probses = torch.cat(probses, 0)
        activationses = torch.cat(activationses, 0)
        return probses.cpu().detach().numpy(), activationses.cpu().detach().numpy()


class FIDandIS(InceptionMetric):
    def __init__(self, inception=None, device=torch.device("cpu"), is_splits=10, **kw):
        super(FIDandIS, self).__init__(inception=inception, device=device)
        self.fid = FID(inception=self.inception, device=self.device)
        self.is_score = IS(inception=self.inception, device=self.device, splits=is_splits)

    def set_real_stats_with(self, data):
        self.fid.set_real_stats_with(data)

    def get_scores(self, data):
        """
        :param data:    dataloader
        :return:
        """
        ises = -1
        fids = -1
        try:
            probs, acts = self.get_inception_outs(data)
            ises = self.is_score.get_scores_from_probs(probs)
            fids = self.fid.get_distance_from_activations(acts)
        except Exception as e:
            print(e)
        return ises, fids

    def __call__(self, data):
        """
        :param data:    dataloader
        :return:
        """
        return self.get_scores(data)


class FID(InceptionMetric):

    def get_data_activations(self, data):
        # self.inception.eval()
        probs, activations = self.get_inception_outs(data)
        return activations

    def get_activation_stats(self, activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def get_distance_from_data(self, gen_data, real_data=None, eps=1e-6):
        acts1 = self.get_data_activations(gen_data)
        acts2 = self.get_data_activations(real_data) if real_data is not None else None

        return self.get_distance_from_activations(acts1, acts2, eps=eps)

    def get_distance_from_activations(self, gen_acts, real_acts=None, eps=1e-6):
        mu1, sigma1 = self.get_activation_stats(gen_acts)
        mu2, sigma2 = self.get_activation_stats(real_acts) if real_acts is not None else self.real_stats

        tt = q.ticktock("scorer")
        tt.tick("computing fid")
        # from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        tt.tock("fid computed")
        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def set_real_stats_with(self, data):        # dataloader
        acts = self.get_data_activations(data)
        mu, sigma = self.get_activation_stats(acts)
        self.real_stats = (mu, sigma)

    def __call__(self, data):
        return self.get_distance_from_data(data)


class IS(InceptionMetric):

    def __init__(self, inception=None, device=torch.device("cpu"), splits=10):
        super(IS, self).__init__(inception=inception, device=device)
        self.splits = splits

    def get_scores_from_data(self, data):     # dataloader
        # self.inception.eval()
        allprobs, _ = self.get_inception_outs(data)
        return self.get_scores_from_probs(allprobs)

    def get_scores_from_probs(self, allprobs):
        tt = q.ticktock("scorer")

        tt.tick("calculating scores")

        scores = []
        splits = self.splits
        for i in range(splits):
            part = allprobs[(i * allprobs.shape[0] // splits):((i + 1) * allprobs.shape[0] // splits), :]
            part_means = np.expand_dims(np.mean(part, 0), 0)
            kl = part * (np.log(part) - np.log(part_means))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))

        tt.tock("calculated scores")
        return np.mean(scores), np.std(scores)

    def __call__(self, data):
        return self.get_scores_from_data(data)


# region tensorflow IS
import tensorflow as tf
import os, sys, functools
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
import time


tfgan = tf.contrib.gan
# INCEPTION_V3_SAVE_PATH="/data/lukovnik/inception_v3"


class tfIS(object):
    def __init__(self, batsize=64, inception_path="none", inception_version="v3", image_size=299, gpu=None):
        super(tfIS, self).__init__()
        self.batsize = batsize
        self.inception_path = inception_path
        if gpu is None:
            self.session = tf.InteractiveSession()
        else:
            config = tf.ConfigProto(device_count={'GPU': gpu})
            self.session = tf.InteractiveSession(config=config)
        self.inpvar = tf.placeholder(tf.float32, [self.batsize, 3, None, None])
        self.inception_version = inception_version
        self.image_size = image_size

        self.logits = self.inception_logits(self.inpvar)

    def prepare_images(self, images):
        images = tf.transpose(images, [0, 2, 3, 1])
        images = tf.image.resize_bilinear(images, [self.image_size, self.image_size])
        return images

    def get_inception_probs(self, inps):
        preds = []
        n_batches = len(inps) // self.batsize
        for i in range(n_batches):
            sys.stdout.write("."); sys.stdout.flush()
            inp = inps[i * self.batsize:(i + 1) * self.batsize]
            pred = self.logits.eval({self.inpvar: inp})[:, :1000]
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
        return preds

    def inception_logits(self, inpvar, num_splits=1):
        images = self.prepare_images(inpvar)
        generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
        if self.inception_version == "default":
            _fn = functools.partial(tfgan.eval.run_inception, output_tensor='logits:0')
        else:
            if self.inception_version == "v1":
                inception_url = "https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz"
                inception_file = "inception_v1_2016_08_28_frozen.pb"
                inception_path = os.path.join(self.inception_path, "inception_v1.pb")
            elif self.inception_version == "v2":
                inception_url = "https://storage.googleapis.com/download.tensorflow.org/models/inception_v2_2016_08_28_frozen.pb.tar.gz"
                inception_file = "inception_v2_2016_08_28_frozen.pb"
                inception_path = os.path.join(self.inception_path, "inception_v2.pb")
            elif self.inception_version == "v3":
                inception_url = "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz"
                inception_file = "inception_v3_2016_08_28_frozen.pb"
                inception_path = os.path.join(self.inception_path, "inception_v3.pb")
                inception_outvar = "import/InceptionV3/Logits:0"
            else:
                raise q.SumTingWongException("unknown inception version {}".format(self.inception_version))

            graphfn = tfgan.eval.get_graph_def_from_url_tarball(
                inception_url,
                inception_file,
                inception_path)
            _fn = functools.partial(tfgan.eval.run_inception, graph_def=graphfn, output_tensor=inception_outvar)#'logits:0')

        logits = functional_ops.map_fn(
            fn=_fn,
            elems=array_ops.stack(generated_images_list),
            parallel_iterations=1,
            back_prop=False,
            swap_memory=True,
            name='RunClassifier')
        logits = array_ops.concat(array_ops.unstack(logits), 0)
        return logits

    def get_inception_score(self, images, splits=10):
        assert (type(images) == np.ndarray)
        assert (len(images.shape) == 4)
        assert (images.shape[1] == 3)
        assert (np.max(images[0]) <= 1)
        assert (np.min(images[0]) >= -1)

        start_time = time.time()
        preds = self.get_inception_probs(images)
        print('Inception Score for %i samples in %i splits' % (preds.shape[0], splits))
        mean, std = self.preds2score(preds, splits)
        print('Inception Score calculation time: %f s' % (time.time() - start_time))
        return mean, std  # Reference values: 11.34 for 49984 CIFAR-10 training set images, or mean=11.31, std=0.08 if in 10 splits (default).

    def preds2score(self, preds, splits):
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)

# endregion



