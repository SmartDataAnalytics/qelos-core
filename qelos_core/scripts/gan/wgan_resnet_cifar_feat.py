import qelos_core as q
import torch
import torchvision
import numpy as np

# region new model
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

# endregion


class SubVGG(torch.nn.Module):
    def __init__(self, version=11, feat_layer=9, pretrained=True, **kw):
        """
        Pretrained VGG-*version*, taking only first *feat_layer*-th layer's output.
        :param version:     11/13/16/19
        """
        super(SubVGG, self).__init__(**kw)
        v2f = {
            11: torchvision.models.vgg11,
            13: torchvision.models.vgg13,
            16: torchvision.models.vgg16,
            19: torchvision.models.vgg19,
        }
        if version not in v2f:
            raise q.SumTingWongException("vgg{} does not exist, please specify valid version number (11, 13, 16 or 19)".format(version))
        self.vgg = v2f[version](pretrained=pretrained)
        if feat_layer > len(self.vgg.features):
            raise q.SumTingWongException("vgg{} does not have layer nr. {}. Please use a valid layer number."
                                         .format(version, feat_layer))
        self.layers = self.vgg.features[:feat_layer]
        def get_numth(num):
            numther = {1: "st", 2: "nd", 3: "rd"}
            if num in numther:
                return numther[num]
            else:
                return "th"
        print("using VGG{}'s {}{} layer's outputs ({})".format(version, feat_layer, get_numth(feat_layer), str(self.layers[feat_layer-1])))

    def forward(self, x):
        feats = self.layers(x)
        return feats


def tst_subvgg():
    v = 13
    l = 8
    vgg = SubVGG(version=v, feat_layer=l)
    x = torch.rand(1, 3, 32, 32) * 2 - 1
    y = vgg(x)
    print(y.size())
    return y.size(1)


def get_vgg_outdim(v, l):
    vgg = SubVGG(version=v, feat_layer=l)
    x = torch.rand(1, 3, 32, 32) * 2 - 1
    y = vgg(x)
    return y.size(1)


def tst_subvgg_with_disc():

    v = 13
    l = 8
    vgg = SubVGG(version=v, feat_layer=l)
    d = OldDiscriminator(128, 128)
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    y = vgg(x)
    z = d(y)
    print(y.size(), z.size(), z)


# region oldmodel
class Normalize(torch.nn.Module):
    def __init__(self, dim, **kw):
        super(Normalize, self).__init__(**kw)
        self.bn = torch.nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(x)


class ConvMeanPool(torch.nn.Module):
    def __init__(self, indim, outdim, filter_size, biases=True, **kw):
        super(ConvMeanPool, self).__init__(**kw)
        assert(filter_size % 2 == 1)
        padding = filter_size // 2
        self.conv = torch.nn.Conv2d(indim, outdim, kernel_size=filter_size, padding=padding, bias=biases)
        self.pool = torch.nn.AvgPool2d(2)

    def forward(self, x):
        y = self.conv(x)
        y = self.pool(y)
        return y


class MeanPoolConv(ConvMeanPool):
    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y)
        return y


class UpsampleConv(torch.nn.Module):
    def __init__(self, indim, outdim, filter_size, biases=True, **kw):
        super(UpsampleConv, self).__init__(**kw)
        assert(filter_size % 2 == 1)
        padding = filter_size // 2
        self.conv = torch.nn.Conv2d(indim, outdim, kernel_size=filter_size, padding=padding, bias=biases)
        self.pool = torch.nn.Upsample(scale_factor=2)

    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y)
        return y


class ResidualBlock(torch.nn.Module):
    def __init__(self, indim, outdim, filter_size, resample=None, use_bn=False, **kw):
        super(ResidualBlock, self).__init__(**kw)
        assert(filter_size % 2 == 1)
        padding = filter_size // 2
        bn2dim = outdim
        if resample == "down":
            self.conv1 = torch.nn.Conv2d(indim, indim, kernel_size=filter_size, padding=padding, bias=True)
            self.conv2 = ConvMeanPool(indim, outdim, filter_size=filter_size)
            self.conv_shortcut = ConvMeanPool
            bn2dim = indim
        elif resample == "up":
            self.conv1 = UpsampleConv(indim, outdim, filter_size=filter_size)
            self.conv2 = torch.nn.Conv2d(outdim, outdim, kernel_size=filter_size, padding=padding, bias=True)
            self.conv_shortcut = UpsampleConv
        else:   # None
            assert(resample is None)
            self.conv1 = torch.nn.Conv2d(indim, outdim, kernel_size=filter_size, padding=padding, bias=True)
            self.conv2 = torch.nn.Conv2d(outdim, outdim, kernel_size=filter_size, padding=padding, bias=True)
            self.conv_shortcut = torch.nn.Conv2d
        if use_bn:
            self.bn1 = Normalize(indim)
            self.bn2 = Normalize(bn2dim)
        else:
            self.bn1, self.bn2 = None, None

        self.nonlin = torch.nn.ReLU()

        if indim == outdim and resample == None:
            self.conv_shortcut = None
        else:
            self.conv_shortcut = self.conv_shortcut(indim, outdim, filter_size=1)       # bias is True by default, padding is 0 by default

    def forward(self, x):
        if self.conv_shortcut is None:
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
        y = self.bn1(x) if self.bn1 is not None else x
        y = self.nonlin(y)
        y = self.conv1(y)
        y = self.bn2(y) if self.bn2 is not None else y
        y = self.nonlin(y)
        y = self.conv2(y)

        return y + shortcut


class OptimizedResBlockDisc1(torch.nn.Module):
    def __init__(self, dim, **kw):
        super(OptimizedResBlockDisc1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, dim, kernel_size=3, padding=1, bias=True)
        self.conv2 = ConvMeanPool(dim, dim, filter_size=3, biases=True)
        self.conv_shortcut = MeanPoolConv(3, dim, filter_size=1, biases=True)
        self.nonlin = torch.nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.nonlin(y)
        y = self.conv2(y)
        shortcut = self.conv_shortcut(x)
        return y + shortcut


class OldGenerator(torch.nn.Module):
    def __init__(self, z_dim, dim_g, use_bn=True, extra_layers=False, **kw):
        super(OldGenerator, self).__init__(**kw)
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, 4*4*dim_g),
            q.Lambda(lambda x: x.view(x.size(0), dim_g, 4, 4)),
            ResidualBlock(dim_g, dim_g, 3, resample="up", use_bn=use_bn),
            ResidualBlock(dim_g, dim_g, 3, resample="up", use_bn=use_bn),
            ResidualBlock(dim_g, dim_g, 3, resample="up", use_bn=use_bn),
        ])
        if extra_layers:
            self.layers.append(ResidualBlock(dim_g, dim_g, 3, resample=None, use_bn=use_bn))
            self.layers.append(ResidualBlock(dim_g, dim_g, 3, resample=None, use_bn=use_bn))
        self.layers.extend([Normalize(dim_g),
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim_g, 3, kernel_size=3, padding=1),
            torch.nn.Tanh(),
        ])

    def forward(self, a):
        for layer in self.layers:
            a = layer(a)
        return a


class OldDiscriminator(torch.nn.Module):
    def __init__(self, inp_d, dim_d, use_bn=False, **kw):
        super(OldDiscriminator, self).__init__(**kw)
        self.layers = torch.nn.ModuleList([
            # OptimizedResBlockDisc1(dim_d),
            ResidualBlock(inp_d, dim_d, 3, resample="down", use_bn=use_bn),
            ResidualBlock(dim_d, dim_d, 3, resample="down", use_bn=use_bn),
            ResidualBlock(dim_d, dim_d, 3, resample=None, use_bn=use_bn),
            torch.nn.ReLU(),
            q.Lambda(lambda x: x.mean(3).mean(2)),
            torch.nn.Linear(dim_d, 1),
            q.Lambda(lambda x: x.squeeze(1))
        ])

    def forward(self, a):
        for layer in self.layers:
            a = layer(a)
        return a
# endregion


class UnquantizeTransform(object):
    def __init__(self, levels=256, range=(-1., 1.)):
        super(UnquantizeTransform, self).__init__()
        self.rand_range = (range[1] - range[0]) * 1. / (1. * levels)

    def __call__(self, x):
        rand = torch.rand_like(x) * self.rand_range
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
        vggversion=13,
        vgglayer=9,
        vggvanilla=False,           # if True, makes trainable feature transform
        extralayers=False,          # adds a couple extra res blocks to generator to match added VGG
        pixelpenalty=False,         # if True, uses penalty based on pixel-wise interpolate
        inceptionpath="/data/lukovnik/",
        ):
        # vggvanilla=True and pixelpenalty=True makes a normal WGAN

    settings = locals().copy()
    logger = q.log.Logger(prefix="wgan_resnet_cifar_feat")
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
    gen = OldGenerator(z_dim, dim_g, extra_layers=extralayers).to(device)
    inpd = get_vgg_outdim(vggversion, vgglayer)
    crit = OldDiscriminator(inpd, dim_d).to(device)
    subvgg = SubVGG(vggversion, vgglayer, pretrained=not vggvanilla)
    tt.tock("created networks")

    # test
    # z = torch.randn(3, z_dim).to(device)
    # x = gen(z)
    # s = crit(x)

    # data
    # load cifar
    tt.tick("loading data")
    traincifar, testcifar = load_cifar_dataset(train=True), load_cifar_dataset(train=False)
    print(len(traincifar), len(testcifar))

    gen_data_d = q.gan.gauss_dataset(z_dim, len(traincifar))
    disc_data = q.datacat([traincifar, gen_data_d], 1)

    gen_data = q.gan.gauss_dataset(z_dim)
    gen_data_valid = q.gan.gauss_dataset(z_dim, 50000)

    swd_gen_data = q.gan.gauss_dataset(z_dim, 16384)
    swd_real_data = []
    swd_shape = traincifar[0].size()
    for i in range(16384):
        swd_real_data.append(traincifar[i])
    swd_real_data = torch.stack(swd_real_data, 0)

    disc_data = q.dataload(disc_data, batch_size=batsize, shuffle=True)
    gen_data = q.dataload(gen_data, batch_size=batsize, shuffle=True)
    gen_data_valid = q.dataload(gen_data_valid, batch_size=batsize, shuffle=False)
    validcifar_loader = q.dataload(testcifar, batch_size=batsize, shuffle=False)

    swd_batsize = 1024
    swd_gen_data = q.dataload(swd_gen_data, batch_size=swd_batsize, shuffle=False)
    swd_real_data = q.dataload(swd_real_data, batch_size=swd_batsize, shuffle=False)

    dev_data_gauss = q.gan.gauss_dataset(z_dim, len(testcifar))
    dev_disc_data = q.datacat([testcifar, dev_data_gauss], 1)
    dev_disc_data = q.dataload(dev_disc_data, batch_size=batsize, shuffle=False)
    # q.embed()
    tt.tock("loaded data")

    disc_model = q.gan.WGAN_F(crit, gen, subvgg, lamda=lamda, pixel_penalty=pixelpenalty).disc_train()
    gen_model = q.gan.WGAN_F(crit, gen, subvgg, lamda=lamda, pixel_penalty=pixelpenalty).gen_train()

    disc_params = q.params_of(crit)
    if vggvanilla:
        disc_params += q.params_of(subvgg)
    disc_optim = torch.optim.Adam(disc_params, lr=lr, betas=(0.5, 0.9))
    gen_optim = torch.optim.Adam(q.params_of(gen), lr=lr, betas=(0.5, 0.9))

    disc_bt = UnquantizeTransform()

    disc_trainer = q.trainer(disc_model).on(disc_data).optimizer(disc_optim).loss(3).device(device)\
        .set_batch_transformer(lambda a, b: (disc_bt(a), b))
    gen_trainer = q.trainer(gen_model).on(gen_data).optimizer(gen_optim).loss(1).device(device)

    # fidandis = q.gan.FIDandIS(device=device)
    tfis = q.gan.tfIS(inception_path=inceptionpath, gpu=gpu)
    # if not test:
    #     fidandis.set_real_stats_with(validcifar_loader)
    saver = q.gan.GenDataSaver(logger, "saved.npz")
    generator_validator = q.gan.GeneratorValidator(gen, [tfis, saver], gen_data_valid, device=device,
                                         logger=logger, validinter=validinter)

    train_validator = q.tester(disc_model).on(dev_disc_data).loss(3).device(device)\
        .set_batch_transformer(lambda a, b: (disc_bt(a), b))

    train_validator.validinter = devinter

    tt.tick("initializing SWD")
    swd = q.gan.SlicedWassersteinDistance(swd_shape)
    swd.prepare_reals(swd_real_data)
    tt.tock("SWD initialized")

    swd_validator = q.gan.GeneratorValidator(gen, [swd], swd_gen_data, device=device,
                                             logger=logger, validinter=validinter, name="swd")

    tt.tick("training")
    gan_trainer = q.gan.GANTrainer(disc_trainer, gen_trainer,
                                   validators=(generator_validator, train_validator, swd_validator),
                                   lr_decay=True)

    gan_trainer.run(epochs, disciters=disciters, geniters=1, burnin=burnin)
    tt.tock("trained")


if __name__ == "__main__":
    # tst_subvgg()
    # tst_subvgg_with_disc()
    q.argprun(run)