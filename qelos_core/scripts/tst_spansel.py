import torch
import qelos_core as q
import numpy as np
import random

""" Dummy test to see if span attention followed by encoder can be trained softly """

def run(lr=0.001,
        seqlen=6,
        numex=200,
        batsize=10):
    # region construct data
    colors = "red blue green magenta cyan orange yellow grey salmon pink purple teal".split()
    D = dict(zip(colors, range(len(colors))))
    inpseqs = []
    targets = []
    for i in range(numex):
        inpseq = list(np.random.choice(colors, seqlen, replace=False))
        target = np.random.choice(range(len(inpseq)), 1)[0]
        target_class = D[inpseq[target]]
        inpseq[target] = "${}$".format(inpseq[target])
        inpseqs.append("".join(inpseq))
        targets.append(target_class)

    sm = q.StringMatrix()
    sm.tokenize = lambda x: list(x)

    for inpseq in inpseqs:
        sm.add(inpseq)

    sm.finalize()
    print(sm[0])
    print(sm.D)
    targets = np.asarray(targets)

    data = q.dataload(sm.matrix, targets, batch_size=batsize)
    # endregion

    # region model
    emb = q.WordEmb(20, worddic=sm.D)       # sm dictionary (characters)
    out = q.WordLinout(20, worddic=D)       # target dictionary
    # encoders:
    enc1 = q.SimpleLSTMEncoder(20, 20, bidir=True)
    enc2 = q.SimpleLSTMEncoder(20, 20, bidir=True)
    # model
    class Model(torch.nn.Module):
        def __init__(self, dim, _emb, _out, _enc1, _enc2, **kw):
            super(Model, self).__init__(**kw)
            self.dim, self.emb, self.out, self.enc1, self.enc2 = dim, _emb, _out, _enc1, _enc2
            self.score = torch.nn.Sequential(
                torch.nn.Linear(dim, 1, bias=False),
                torch.nn.Sigmoid())

        def forward(self, x):
            # embed and encode
            xemb, xmask = self.emb(x)
            xenc = self.enc1(xemb, mask=xmask)
            # compute attention
            xatt = self.score(xenc)
            # encode again
            return None

    model = Model(40, emb, out, enc1, enc2)
    # endregion

    # region test
    outs = model(torch.tensor(sm.matrix[0:2]))
    # endregion





if __name__ == '__main__':
    q.argprun(run)