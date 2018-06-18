import torch
import qelos_core as q


class Decoder(torch.nn.Module):
    """ self-sampling decoder """
    def __init__(self, D, maxtime, embdim, *innerdim, **kw):
        super(Decoder, self).__init__()
        self.emb = q.WordEmb(embdim, worddic=D)
        innerdim = (embdim,) + innerdim
        self.layers = torch.nn.ModuleList(modules=[
            torch.nn.LSTMCell(innerdim[i-1], innerdim[i]) for i in range(1, len(innerdim))
        ])


def run(lr=0.001):



if __name__ == "__main__":
    q.argprun(run)