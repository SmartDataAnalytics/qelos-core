import qelos_core as q
import torch
import torchvision
import numpy as np


# region from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class ResamplingConv(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, resample=None):
        super(ResamplingConv, self).__init__()
        assert(kernel in (1, 3, 5, 7))
        padding = (kernel - 1) // 2
        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel, padding=padding, bias=False)
        self.resample = resample
        if resample == "up":
            self.resampler = torch.nn.Upsample(scale_factor=2)
        elif resample == "down":
            self.resampler = torch.nn.AvgPool2d(2)

    def forward(self, x):
        if self.resample == "up":
            x = self.resampler(x)
        x = self.conv(x)
        if self.resample == "down":
            x = self.resampler(x)
        return x


class ResBlock(torch.nn.Module):
    def __init__(self, inplanes, planes, kernel=3, resample=None):
        super(ResBlock, self).__init__()
        self.conv1 = ResamplingConv(inplanes, planes, kernel, resample=resample if resample != "down" else None)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = ResamplingConv(planes, planes, kernel, resample=resample if resample != "up" else None)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = ResamplingConv(inplanes, planes, 1, resample=resample)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.resample is not None:
            residual = self.resample(x)

        out += residual
        out = self.relu(out)

        return out

# endregion


class Generator(torch.nn.Module):
    def __init__(self, z_dim, dim_g, **kw):
        super(Generator, self).__init__(**kw)
        self.layers = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(z_dim, dim_g, 4),
            ResBlock(dim_g, dim_g, 3, resample='up'),
            ResBlock(dim_g, dim_g, 3, resample='up'),
            ResBlock(dim_g, dim_g, 3, resample='up'),
            torch.nn.BatchNorm2d(dim_g),
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim_g, 3, 3),
            torch.nn.Tanh(),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, dim_d, **kw):
        super(Discriminator, self).__init__(**kw)




def load_cifar_dataset():
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)
    cifar = torchvision.datasets.CIFAR10(root='../../datasets/cifar/', download=True,
                         transform=torchvision.transforms.Compose([
                             torchvision.transforms.Scale(32),
                             torchvision.transforms.ToTensor(),
                             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])
                         )
    cifar = IgnoreLabelDataset(cifar)

    cifar_npy = []
    for i in range(len(cifar)):
        cifar_npy.append(cifar[i].unsqueeze(0).numpy())
    cifar_npy = np.concatenate(cifar_npy, 0)

    return cifar_npy


def run(lr=0.0001,
        batsize=64,
        epochs=100000,
        lamda=10,
        disciters=10,
        burnin=500,
        validinter=500,
        cuda=False,
        gpu=0,
        z_dim=64,
        test=False):
    splits = (8, 1, 1)

    settings = locals().copy()
    logger = q.log.Logger(prefix="csigan")
    logger.save_settings(**settings)

    if test:
        validinter=1
        burnin=1
        batsize=2
        splits = (50, 50, 49900)

    tt = q.ticktock("script")

    device = torch.device("cpu") if not cuda else torch.device("cuda", gpu)

    tt.tick("creating networks")
    gen_layers = create_basic_gen(z_dim)
    gen = Generator(gen_layers).to(device)
    crit_layers = create_basic_critic(z_dim)
    crit = Critic(crit_layers).to(device)
    tt.tock("created networks")

    # test
    z = torch.randn(3, z_dim).to(device)
    x = gen(z)
    s = crit(x)

    # data
    # load cifar
    tt.tick("loading data")
    cifar = load_cifar_dataset()
    traincifar, validcifar, testcifar = q.datasplit([cifar], splits=splits, random=True)

    realdata = q.dataset(traincifar)
    gen_data_d = q.gan.gauss_dataset(z_dim, len(realdata))
    disc_data = q.datacat([realdata, gen_data_d], 1)

    gen_data = q.gan.gauss_dataset(z_dim)
    gen_data_valid = q.gan.gauss_dataset(z_dim, len(validcifar[0]))

    disc_data = q.dataload(disc_data, batch_size=batsize, shuffle=True)
    gen_data = q.dataload(gen_data, batch_size=batsize, shuffle=True)
    gen_data_valid = q.dataload(gen_data_valid, batch_size=batsize, shuffle=False)
    validcifar_loader = q.dataload(validcifar[0], batch_size=batsize, shuffle=False)
    # q.embed()
    tt.tock("loaded data")

    disc_model = q.gan.WGAN(crit, gen, lamda=lamda).disc_train()
    gen_model = q.gan.WGAN(crit, gen, lamda=lamda).gen_train()

    disc_optim = torch.optim.Adam(q.params_of(crit), lr=lr)
    gen_optim = torch.optim.Adam(q.params_of(gen), lr=lr)

    disc_trainer = q.trainer(disc_model).on(disc_data).optimizer(disc_optim).loss(3).device(device)
    gen_trainer = q.trainer(gen_model).on(gen_data).optimizer(gen_optim).loss(1).device(device)

    fidandis = q.gan.FIDandIS(device=device)
    fidandis.set_real_stats_with(validcifar_loader)
    saver = q.gan.GenDataSaver(logger, "saved.npz")
    validator = q.gan.Validator(gen, [fidandis, saver], gen_data_valid, device=device, logger=logger)

    tt.tick("training")
    gan_trainer = q.gan.GANTrainer(disc_trainer, gen_trainer, validator=validator)

    gan_trainer.run(epochs, disciters=disciters, geniters=1, burnin=burnin, validinter=validinter)
    tt.tock("trained")


if __name__ == "__main__":
    q.argprun(run)