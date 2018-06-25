import qelos_core as q
import torch
import numpy as np
import math


### PROGRESSIVE GAN #######


class StagedGenerator(torch.nn.Module):
    def __init__(self, layers, **kw):
        """
        :param layers:  the blocks handling a certain resolution, every next layer must return x2 resolution,
                        must contain a .torgb attribute with module in it
        :param torgbs:  map the output of a certain block to rgb, must match number of layers
        :param kw:
        """
        super(StagedGenerator, self).__init__()
        self.layers = torch.nn.ModuleList(modules=layers)

    def forward(self, z, stage=-1):
        """
        :param z:       sample from source distribution
        :param stage:   where to stop, -1 = end, int = whole, float=partial (as in Progressive GAN paper)
        :return:        generated sample at ceil(stage)'s layer resolution
        """
        stage = stage * 1.
        ret = z.unsqueeze(2).unsqueeze(3)
        prevret = ret
        i = 0
        for i in range(len(self.layers)):
            prevret = ret
            ret = self.layers[i](ret)
            if i >= math.ceil(stage) and stage >= 0:
                break
        if stage % 1.0 == 0 or stage < 0:
            ret = self.layers[i].torgb(ret)
        else:
            ret = torch.nn.functional.upsample(
                        self.layers[int(math.floor(stage))].torgb(prevret), scale_factor=2)\
                            * (math.ceil(stage) - stage) \
                  + self.layers[int(math.ceil(stage))].torgb(ret) \
                            * (stage - math.floor(stage))
        return ret


class StagedDiscriminator(torch.nn.Module):
    def __init__(self, layers, **kw):
        super(StagedDiscriminator, self).__init__()
        self.layers = torch.nn.ModuleList(modules=layers)

    def forward(self, x, stage=-1):
        stage = stage * 1.
        ret = x
        prevret = ret
        for i in range(len(self.layers)):
            if len(self.layers) - i - 1 > stage:
                prevret = ret
                ret = self.layers[i](ret)




class GenConvBlock(torch.nn.Module):
    def __init__(self, scale, channels, dims, paddings, leakiness=0.2, **kw):
        super(GenConvBlock, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Upsample(scale_factor=scale))
        for i in range(len(dims)):
            dim, padding = dims[i], paddings[i]
            inp_chan, out_chan = channels[i], channels[i+1]
            self.layers.append(torch.nn.Conv2d(inp_chan, out_chan, dim, padding=padding))
            self.layers.append(torch.nn.LeakyReLU(leakiness))
        self.torgb = torch.nn.Conv2d(channels[-1], 3, 1)    # 1x1 convolution to RGB

    def forward(self, x):
        ret = x
        _x = x
        for layer in self.layers:
            ret = layer(ret)
        return ret


def create_gen_1kx1k_basic(z_dim=512):
    layers = []
    layers.append(GenConvBlock(1, (z_dim, z_dim), (3,), (1,)))
    layers[-1].layers[0] = torch.nn.ConvTranspose2d(z_dim, z_dim, 4)
    layers.append(GenConvBlock(2, (z_dim, z_dim, z_dim), (3, 3), (1, 1)))
    layers.append(GenConvBlock(2, (z_dim, z_dim, z_dim), (3, 3), (1, 1)))
    layers.append(GenConvBlock(2, (z_dim, z_dim, z_dim), (3, 3), (1, 1)))
    layers.append(GenConvBlock(2, (z_dim, z_dim // 2, z_dim // 2), (3, 3), (1, 1)))
    layers.append(GenConvBlock(2, (z_dim // 2, z_dim // 4, z_dim // 4), (3, 3), (1, 1)))
    layers.append(GenConvBlock(2, (z_dim // 4, z_dim // 8, z_dim // 8), (3, 3), (1, 1)))
    layers.append(GenConvBlock(2, (z_dim // 8, z_dim // 16, z_dim // 16), (3, 3), (1, 1)))
    layers.append(GenConvBlock(2, (z_dim // 16, z_dim // 32, z_dim // 32), (3, 3), (1, 1)))

    return layers



def run(lr=0.001,
        z_dim=64):

    # generator:
    gen_layers = create_gen_1kx1k_basic(z_dim)
    gen = StagedGenerator(gen_layers)

    # test
    z = torch.randn(3, z_dim)
    x = gen(z, stage=1.5)


if __name__ == "__main__":
    q.argprun(run)