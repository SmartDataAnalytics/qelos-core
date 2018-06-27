import qelos_core as q
import torch
import torchvision
import numpy as np
import math


### PROGRESSIVE GAN #######


class Generator(torch.nn.Module):
    def __init__(self, layers, **kw):
        """
        :param layers:  the blocks handling a certain resolution, every next layer must return x2 resolution,
                        must contain a .torgb attribute with module in it
        :param torgbs:  map the output of a certain block to rgb, must match number of layers
        :param kw:
        """
        super(Generator, self).__init__()
        self.layers = torch.nn.ModuleList(modules=layers)

    def forward(self, z):
        """
        :param z:       sample from source distribution
        """
        z = z.unsqueeze(2).unsqueeze(3)
        x, rgb = self.layers[0](z)
        for i in range(1, len(self.layers)):
            x, rgb = self.layers[i](x, rgb)
        return rgb


class ZeroGenBlock(torch.nn.Module):
    def __init__(self, z_dim, window=4, leakiness=0.2, **kw):
        super(ZeroGenBlock, self).__init__(**kw)
        self.conv = torch.nn.ConvTranspose2d(z_dim, z_dim, window)
        self.relu = torch.nn.LeakyReLU(leakiness)
        self.torgb = torch.nn.Conv2d(z_dim, 3, 1)

    def forward(self, z):
        x = self.conv(z)
        x = self.relu(x)
        rgb = self.torgb(x)
        return x, rgb


class GenConvBlock(torch.nn.Module):
    def __init__(self, scale, inp_channels, channels, dims, paddings, leakiness=0.2, **kw):
        super(GenConvBlock, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.scale = scale
        channels = (inp_channels+3,) + channels
        for i in range(len(dims)):
            dim, padding = dims[i], paddings[i]
            inp_chan, out_chan = channels[i], channels[i+1]
            self.layers.append(torch.nn.Conv2d(inp_chan, out_chan, dim, padding=padding))
            self.layers.append(torch.nn.LeakyReLU(leakiness))
        self.torgb = torch.nn.Conv2d(channels[-1], 3, 1)    # 1x1 convolution to RGB

    def forward(self, x, prevrgb):
        _x = x
        _prgb = prevrgb
        if self.scale > 1:
            _x = torch.nn.functional.upsample(_x, scale_factor=self.scale, mode="bilinear")
            _prgb = torch.nn.functional.upsample(_prgb, scale_factor=self.scale, mode="bilinear")
        _x = torch.cat([_x, _prgb], 1)

        for layer in self.layers:
            _x = layer(_x)
        rgb = self.torgb(_x)

        rgb = _prgb + rgb
        #rgb = rgb.clamp(-1, 1)

        return _x, rgb


class Critic(torch.nn.Module):
    def __init__(self, layers, **kw):
        super(Critic, self).__init__(**kw)
        self.layers = torch.nn.ModuleList(modules=layers)

    def forward(self, x):
        _x = x
        _rgb = None
        for i in range(len(self.layers)):
            _x, _rgb = self.layers[i](_x, _rgb)
        return _x


class CriticBlock(torch.nn.Module):
    def __init__(self, scale=1, inp_channels=0, channels=None, dims=None, paddings=None, leakiness=0.2, **kw):
        super(CriticBlock, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.scale = scale
        self.inp_channels = inp_channels
        channels = (inp_channels + 3,) + channels
        for i in range(len(dims)):
            dim, padding = dims[i], paddings[i]
            inp_chan, out_chan = channels[i], channels[i+1]
            self.layers.append(torch.nn.Conv2d(inp_chan, out_chan, dim, padding=padding))
            self.layers.append(torch.nn.LeakyReLU(leakiness))

    def forward(self, x, rgb=None):
        _x = x if self.inp_channels == 0 else torch.cat([x, rgb], 1)
        _rgb = x if self.inp_channels == 0 else rgb
        for layer in self.layers:
            _x = layer(_x)
        if self.scale > 1:
            _x = torch.nn.functional.avg_pool2d(_x, self.scale)
            _rgb = torch.nn.functional.avg_pool2d(_rgb, self.scale)
        return _x, _rgb


class LastCriticBlock(torch.nn.Module):
    def __init__(self, dim, leakiness=0.2, **kw):
        super(LastCriticBlock, self).__init__(**kw)
        # self.interlin = torch.nn.Linear(dim, dim, bias=True)
        # self.interlin_act = torch.nn.LeakyReLU(leakiness)
        self.lin = torch.nn.Linear(dim, 1, bias=True)

    def forward(self, x, rgb=None):
        _x = x.squeeze(3).squeeze(2)
        assert(_x.dim() == 2)
        # _x = self.interlin(_x)
        # _x = self.interlin_act(_x)
        _x = self.lin(_x)
        _x = _x.squeeze(1)
        assert(_x.dim() == 1)
        return _x, None


def create_basic_gen(z_dim=64):
    layers = []
    layers.append(ZeroGenBlock(z_dim))
    layers.append(GenConvBlock(1, z_dim, (z_dim, z_dim), (3, 3), (1, 1)))
    layers.append(GenConvBlock(2, z_dim, (z_dim, z_dim), (3, 3), (1, 1)))
    layers.append(GenConvBlock(2, z_dim, (z_dim, z_dim), (3, 3), (1, 1)))
    layers.append(GenConvBlock(2, z_dim, (z_dim//2, z_dim//2), (3, 3), (1, 1)))
    layers.append(GenConvBlock(1, z_dim//2, (z_dim//4, z_dim//4), (3, 3), (1, 1)))
    layers.append(GenConvBlock(1, z_dim//4, (z_dim//8, z_dim//8), (3, 3), (1, 1)))
    return layers


def create_basic_critic(z_dim=64):
    layers = []
    layers.append(CriticBlock(1, 0, (z_dim//8, z_dim//8, z_dim//8), (1, 3, 3), (0, 1, 1)))
    layers.append(CriticBlock(1, z_dim//8, (z_dim//4, z_dim//4), (3, 3), (1, 1)))
    layers.append(CriticBlock(2, z_dim//4, (z_dim//2, z_dim//2), (3, 3), (1, 1)))
    layers.append(CriticBlock(2, z_dim//2, (z_dim, z_dim), (3, 3), (1, 1)))
    layers.append(CriticBlock(2, z_dim, (z_dim, z_dim), (3, 3), (1, 1)))
    layers.append(CriticBlock(1, z_dim, (z_dim, z_dim), (4,), (0,)))
    layers.append(LastCriticBlock(z_dim))
    return layers


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
        lamda=5,
        disciters=5,
        cuda=False,
        gpu=0,
        z_dim=64):
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
    traincifar, validcifar, testcifar = q.datasplit([cifar], splits=(8, 1, 2), random=True)

    realdata = q.dataset(traincifar)
    gen_data_d = q.gan.gauss_dataset(z_dim, len(realdata))
    disc_data = q.datacat([realdata, gen_data_d], 1)

    gen_data = q.gan.gauss_dataset(z_dim)
    tt.tock("loaded data")

    disc_data = q.dataload(disc_data, batch_size=batsize, shuffle=True)
    gen_data = q.dataload(gen_data, batch_size=batsize, shuffle=True)

    disc_model = q.gan.WGAN(crit, gen, lamda=lamda).disc_train()
    gen_model = q.gan.WGAN(crit, gen, lamda=lamda).gen_train()

    disc_optim = torch.optim.Adam(q.params_of(crit), lr=lr)
    gen_optim = torch.optim.Adam(q.params_of(gen), lr=lr)

    disc_trainer = q.trainer(disc_model).on(disc_data).optimizer(disc_optim).loss(3).device(device)
    gen_trainer = q.trainer(gen_model).on(gen_data).optimizer(gen_optim).loss(1).device(device)

    gan_trainer = q.gan.GANTrainer(disc_trainer, gen_trainer)

    tt.tick("training")
    gan_trainer.run(epochs, disciters=disciters, geniters=1, burnin=500)
    tt.tock("trained")


if __name__ == "__main__":
    q.argprun(run)