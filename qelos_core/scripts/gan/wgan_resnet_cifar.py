import qelos_core as q
import torch
import torchvision
import numpy as np


# region from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class ResamplingConv(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, padding=None, resample=None, bias=True):
        """
        Combines resampling (up/down) with convolution.
        If resample is "up", then nn.Upsample(x2) is applied before the conv
        If resample is "down", then nn.MaxPool2D(x2) is applied after the conv
        Padding is automatically set to preserve spatial shapes of input.
        If kernel is 0, no conv is applied and the module is reduced to up/down sampling (if any)
        Resample=None and kernel=0  ==>  nothing happens
        """
        super(ResamplingConv, self).__init__()
        assert(kernel in (0, 1, 3, 5, 7))
        padding = (kernel - 1) // 2 if padding is None else padding
        if kernel == 0:
            self.conv = None
        else:
            self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel, padding=padding, bias=bias)
        self.resample = resample
        if resample == "up":
            self.resampler = torch.nn.Upsample(scale_factor=2)
        elif resample == "down":
            self.resampler = torch.nn.AvgPool2d(2)

    def forward(self, x):
        if self.resample == "up":
            x = self.resampler(x)
        if self.conv is not None:
            x = self.conv(x)
        if self.resample == "down":
            x = self.resampler(x)
        return x


class ResBlock(torch.nn.Module):
    def __init__(self, inplanes, planes, kernel=3, resample=None, bias=True, batnorm=True):
        """
        Residual block with two convs and optional resample.
        If resample == "up", the first conv is upsampling by 2, residual is upsampled too
        If resample == "down", the last conv is downsampling by 2, residual is downsampled too
        If resample == None, no resampling anywhere

        1x1 conv is applied to residual only if inplanes != planes and resample is None.
        """
        super(ResBlock, self).__init__()
        self.conv1 = ResamplingConv(inplanes, planes, kernel,
                                    resample=resample if resample != "down" else None,
                                    bias=bias)
        self.bn1 = torch.nn.BatchNorm2d(planes) if batnorm else None
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = ResamplingConv(planes, planes, kernel,
                                    resample=resample if resample != "up" else None,
                                    bias=bias)
        self.bn2 = torch.nn.BatchNorm2d(planes) if batnorm else None
        self.resample = resample
        self.shortcut = ResamplingConv(inplanes, planes, 0 if (inplanes == planes and resample is None) else 1,
                                       resample=resample,
                                       bias=bias)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out) if self.bn1 is not None else out
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) if self.bn2 is not None else out

        residual = self.shortcut(residual)

        out += residual
        out = self.relu(out)

        return out

# endregion


class Generator(torch.nn.Module):
    def __init__(self, z_dim, dim_g, **kw):
        super(Generator, self).__init__(**kw)
        self.layers = torch.nn.ModuleList([
            q.Lambda(lambda x: x.unsqueeze(2).unsqueeze(3)),
            torch.nn.ConvTranspose2d(z_dim, dim_g, 4),
            torch.nn.BatchNorm2d(dim_g),
            torch.nn.ReLU(),
            ResBlock(dim_g, dim_g, 3, resample='up'),
            ResBlock(dim_g, dim_g, 3, resample='up'),
            ResBlock(dim_g, dim_g, 3, resample='up'),
            torch.nn.Conv2d(dim_g, 3, 3, padding=1),
            torch.nn.Tanh(),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, dim_d, **kw):
        super(Discriminator, self).__init__(**kw)
        self.layers = torch.nn.ModuleList([
            ResBlock(3, dim_d, 3, resample='down', batnorm=False),
            ResBlock(dim_d, dim_d, 3, resample='down', batnorm=False),
            ResBlock(dim_d, dim_d, 3, resample=None, batnorm=False),
            ResBlock(dim_d, dim_d, 3, resample=None, batnorm=False),
            q.Lambda(lambda x: x.mean(3).mean(2)),      # global average pooling over spatial dims
            torch.nn.Linear(dim_d, 1),
            q.Lambda(lambda x: x.squeeze(1))
        ])

    def forward(self, x):   # (batsize, channels, h?, w?)
        for layer in self.layers:
            x = layer(x)
        return x


class UnquantizeTransform(object):
    def __init__(self, levels=256, range=(-1, 1)):
        super(UnquantizeTransform, self).__init__()
        self.rand_range = (range[1] - range[0]) * 1. / (1. * levels)

    def __call__(self, x):
        rand = (torch.rand_like(x) - 0.5) * self.rand_range
        x = x + rand
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(rand_range={0})'.format(self.rand_range)


def load_cifar_dataset(train=True):
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)
    cifar = torchvision.datasets.CIFAR10(root='../../../datasets/cifar/', download=True, train=train,
                         transform=torchvision.transforms.Compose([
                             torchvision.transforms.Scale(32),
                             torchvision.transforms.ToTensor(),
                             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])
                         )
    cifar = IgnoreLabelDataset(cifar)
    return cifar


def run(lr=0.0001,
        batsize=64,
        epochs=100000,
        lamda=10,
        disciters=5,
        burnin=-1,
        validinter=1000,
        devinter=100,
        cuda=False,
        gpu=0,
        z_dim=128,
        test=False,
        dim_d=128,
        dim_g=128,
        ):

    settings = locals().copy()
    logger = q.log.Logger(prefix="resnet_cifar")
    logger.save_settings(**settings)

    burnin = disciters if burnin == -1 else burnin

    if test:
        validinter=10
        burnin=1
        batsize=2
        devinter = 1

    tt = q.ticktock("script")

    device = torch.device("cpu") if not cuda else torch.device("cuda", gpu)

    tt.tick("creating networks")
    gen = Generator(z_dim, dim_g).to(device)
    crit = Discriminator(dim_d).to(device)
    tt.tock("created networks")

    # test
    # z = torch.randn(3, z_dim).to(device)
    # x = gen(z)
    # s = crit(x)

    # data
    # load cifar
    tt.tick("loading data")
    traincifar, testcifar = load_cifar_dataset(train=True), load_cifar_dataset(train=False)
    print(len(traincifar))

    gen_data_d = q.gan.gauss_dataset(z_dim, len(traincifar))
    disc_data = q.datacat([traincifar, gen_data_d], 1)

    gen_data = q.gan.gauss_dataset(z_dim)
    gen_data_valid = q.gan.gauss_dataset(z_dim, 50000)

    disc_data = q.dataload(disc_data, batch_size=batsize, shuffle=True)
    gen_data = q.dataload(gen_data, batch_size=batsize, shuffle=True)
    gen_data_valid = q.dataload(gen_data_valid, batch_size=batsize, shuffle=False)
    validcifar_loader = q.dataload(testcifar, batch_size=batsize, shuffle=False)

    dev_data_gauss = q.gan.gauss_dataset(z_dim, len(testcifar))
    dev_disc_data = q.datacat([testcifar, dev_data_gauss], 1)
    dev_disc_data = q.dataload(dev_disc_data, batch_size=batsize, shuffle=False)
    # q.embed()
    tt.tock("loaded data")

    disc_model = q.gan.WGAN(crit, gen, lamda=lamda).disc_train()
    gen_model = q.gan.WGAN(crit, gen, lamda=lamda).gen_train()

    disc_optim = torch.optim.Adam(q.params_of(crit), lr=lr, betas=(0.5, 0.9))
    gen_optim = torch.optim.Adam(q.params_of(gen), lr=lr, betas=(0.5, 0.9))

    disc_bt = UnquantizeTransform()

    disc_trainer = q.trainer(disc_model).on(disc_data).optimizer(disc_optim).loss(3).device(device)\
        .set_batch_transformer(lambda a, b: (disc_bt(a), b))
    gen_trainer = q.trainer(gen_model).on(gen_data).optimizer(gen_optim).loss(1).device(device)

    fidandis = q.gan.FIDandIS(device=device)
    if not test:
        fidandis.set_real_stats_with(validcifar_loader)
    saver = q.gan.GenDataSaver(logger, "saved.npz")
    generator_validator = q.gan.GeneratorValidator(gen, [fidandis, saver], gen_data_valid, device=device,
                                         logger=logger, validinter=validinter)

    train_validator = q.tester(disc_model).on(dev_disc_data).loss(3).device(device)\
        .set_batch_transformer(lambda a, b: (disc_bt(a), b))

    train_validator.validinter = devinter

    tt.tick("training")
    gan_trainer = q.gan.GANTrainer(disc_trainer, gen_trainer,
                                   validators=(generator_validator, train_validator),
                                   lr_decay=True)

    gan_trainer.run(epochs, disciters=disciters, geniters=1, burnin=burnin)
    tt.tock("trained")


if __name__ == "__main__":
    q.argprun(run)