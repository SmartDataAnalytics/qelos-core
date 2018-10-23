import torch
import qelos_core as q


class Dropout(torch.nn.Module):
    def __init__(self, p=.5, inplace=False):
        super(Dropout, self).__init__()
        self.d = torch.nn.Dropout(p=p, inplace=inplace)
        self.p = p

    def forward(self, *x):
        y = [self.d(z) for z in x]
        y = y[0] if len(y) == 1 else y
        return y


class LayerNormalization(torch.nn.Module):
    ''' Layer normalization module '''

    def __init__(self, dim, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = torch.nn.Parameter(torch.ones(dim), requires_grad=True)
        self.b_2 = torch.nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class NoneLayer(torch.nn.Module):
    def forward(self, *input):
        ret = input
        if len(ret) == 1:
            ret = ret[0]
        return ret


class Lambda(torch.nn.Module):
    def __init__(self, lamda, reg_params=None, reg_modules=None):
        super(Lambda, self).__init__()
        self.lamda = lamda
        if reg_params is not None:
            self._reg_params = torch.nn.ParameterList(parameters=reg_params)
        if reg_modules is not None:
            self._reg_modules = torch.nn.ModuleList(modules=reg_modules)

    def forward(self, x):
        return self.lamda(x)