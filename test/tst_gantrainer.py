import qelos_core as q
import torch
import numpy as np


def run(lr=0.001):
    # data
    x = torch.randn(1000, 5, 5)

    real_data = q.dataset(x)
    gen_data_d = q.gan.gauss_dataset(10, len(real_data))
    disc_data = q.datacat([real_data, gen_data_d], 1)

    gen_data = q.gan.gauss_dataset(10)

    disc_data = q.dataload(disc_data, batch_size=20, shuffle=True)
    gen_data = q.dataload(gen_data, batch_size=20, shuffle=True)

    iter(disc_data).next()

    # models
    class Generator(torch.nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.lin1 = torch.nn.Linear(10, 20)
            self.lin2 = torch.nn.Linear(20, 25)

        def forward(self, z):
            ret = self.lin1(z)
            ret = torch.nn.functional.sigmoid(ret)
            ret = self.lin2(ret)
            ret = torch.nn.functional.sigmoid(ret)
            ret = ret.view(z.size(0), 5, 5)
            return ret

    class Discriminator(torch.nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.lin1 = torch.nn.Linear(25, 20)
            self.lin2 = torch.nn.Linear(20, 10)
            self.lin3 = torch.nn.Linear(10, 1)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            ret = self.lin1(x)
            ret = torch.nn.functional.sigmoid(ret)
            ret = self.lin2(ret)
            ret = torch.nn.functional.sigmoid(ret)
            ret = self.lin3(ret)
            ret = torch.nn.functional.sigmoid(ret)
            ret = ret.squeeze(1)
            return ret

    discriminator = Discriminator()
    generator = Generator()

    disc_model = q.gan.GAN(discriminator, generator, gan_mode=q.gan.GAN.DISC_TRAIN)
    gen_model = q.gan.GAN(discriminator, generator, gan_mode=q.gan.GAN.GEN_TRAIN)

    disc_optim = torch.optim.Adam(q.params_of(discriminator), lr=lr)
    gen_optim = torch.optim.Adam(q.params_of(generator), lr=lr)

    disc_trainer = q.trainer(disc_model).on(disc_data).optimizer(disc_optim).loss(q.no_losses(1))
    gen_trainer = q.trainer(gen_model).on(gen_data).optimizer(gen_optim).loss(q.no_losses(1))

    gan_trainer = q.gan.GANTrainer(disc_trainer, gen_trainer)

    gan_trainer.run(50)



if __name__ == "__main__":
    run()