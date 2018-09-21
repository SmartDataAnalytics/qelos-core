import torch
import qelos_core as q
import numpy as np
import random

""" Dummy test to see if span attention followed by encoder can be trained softly """
class ExpandVecs(torch.nn.Module):
    """ Repeats input vector of 'fromdim' on 'axis' until 'todim' is reached """
    def __init__(self, fromdim, todim, axis):
        super(ExpandVecs, self).__init__()
        self.fromdim, self.todim, self.axis = fromdim, todim, axis
        self.numrepeats = (self.todim // self.fromdim) + 1

    def forward(self, x):
        if self.numrepeats > 1:
            repargs = [1 for _ in x.size()]
            repargs[self.axis] = self.numrepeats
            _x = x.repeat(*repargs)
        else:
            _x = x
        if self.todim != _x.size(self.axis):
            sliceargs = [slice(None, None, None) for _ in x.size()]
            sliceargs[self.axis] = slice(None, self.todim, None)
            _x = _x.__getitem__(sliceargs)
        return _x


def run(lr=0.001,
        seqlen=6,
        numex=500,
        epochs=100,
        batsize=10,
        test=True,
        cuda=False,
        gpu=0):
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda", gpu)
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

    data = q.dataload(sm.matrix[:-100], targets[:-100], batch_size=batsize)
    valid_data = q.dataload(sm.matrix[-100:], targets[-100:], batch_size=batsize)
    # endregion

    # region model
    embdim = 20
    enc2inpdim = 45
    encdim = 20
    outdim = 20
    emb = q.WordEmb(embdim, worddic=sm.D)       # sm dictionary (characters)
    out = q.WordLinout(outdim, worddic=D)       # target dictionary
    # encoders:
    enc1 = q.SimpleLSTMEncoder(embdim, encdim, bidir=True)
    enc2 = q.LSTMCellEncoder(enc2inpdim, outdim, bidir=False)

    # model
    class Model(torch.nn.Module):
        def __init__(self, dim, _emb, _out, _enc1, _enc2, **kw):
            super(Model, self).__init__(**kw)
            self.dim, self.emb, self.out, self.enc1, self.enc2 = dim, _emb, _out, _enc1, _enc2
            self.score = torch.nn.Sequential(
                torch.nn.Linear(dim, 1, bias=False),
                torch.nn.Sigmoid())
            self.emb_expander = ExpandVecs(embdim, enc2inpdim, 2)
            self.enc_expander = ExpandVecs(encdim * 2, enc2inpdim, 2)

        def forward(self, x, with_att=False):
            # embed and encode
            xemb, xmask = self.emb(x)
            xenc = self.enc1(xemb, mask=xmask)
            # compute attention
            xatt = self.score(xenc).squeeze(2) * xmask.float()[:, :xenc.size(1)]
            # encode again
            _xemb = self.emb_expander(xemb[:, :xenc.size(1)])
            _xenc = self.enc_expander(xenc)
            xenc2, _ = self.enc2(_xemb+_xenc, gate=xatt, mask=xmask[:, :xenc.size(1)])
            scores = self.out(xenc2)
            if with_att:
                return scores, xatt
            else:
                return scores

    model = Model(40, emb, out, enc1, enc2)
    # endregion

    # region test
    if test:
        inps = torch.tensor(sm.matrix[0:2])
        outs = model(inps)
    # endregion

    # region train
    optimizer = torch.optim.Adam(q.params_of(model), lr=lr)
    trainer = q.trainer(model).on(data).loss(torch.nn.CrossEntropyLoss(), q.Accuracy())\
        .optimizer(optimizer).hook(q.ClipGradNorm(5.)).device(device)
    validator = q.tester(model).on(valid_data).loss(q.Accuracy()).device(device)
    q.train(trainer, validator).run(epochs=epochs)
    # endregion

    # region check attention    #TODO
    # feed a batch
    inpd = torch.tensor(sm.matrix[400:410])
    outd, att = model(inpd, with_att=True)
    outd = torch.max(outd, 1)[1].cpu().detach().numpy()
    inpd = inpd.cpu().detach().numpy()
    att = att.cpu().detach().numpy()
    rD = {v: k for k, v in sm.D.items()}
    roD = {v: k for k, v in D.items()}
    for i in range(len(att)):
        inpdi = "   ".join([rD[x] for x in inpd[i]])
        outdi = roD[outd[i]]
        print("input:     {}\nprediction: {}\nattention: {}".format(inpdi, outdi,
                                        " ".join(["{:.1f}".format(x) for x in att[i]])))


    # endregion





if __name__ == '__main__':
    q.argprun(run)