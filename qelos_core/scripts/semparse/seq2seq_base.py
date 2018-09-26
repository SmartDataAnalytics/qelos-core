import torch
import qelos_core as q
import os


""" seq2seq baseline on raw data from Dong & Lapata, 2016 """


def load_data(p="../../../datasets/semparse/", which=None, devfrac=0.1, devfracrandom=False):
    tt = q.ticktock("dataloader")
    tt.tick("loading data")
    assert(which is not None)
    which = {"geo": "geoquery", "atis": "atis", "jobs": "jobs"}[which]
    trainp = os.path.join(p, which, "train.txt")
    testp = os.path.join(p, which, "test.txt")
    devp = os.path.join(p, which, "dev.txt")

    trainlines = open(trainp).readlines()
    testlines = open(testp).readlines()

    if not os.path.exists(devp):
        tt.msg("no dev file, taking {} from training data".format(devfrac))
        splitidx = round(len(trainlines)*devfrac)
        trainlines = trainlines[:-splitidx]
        devlines = trainlines[-splitidx:]
    else:
        devlines = open(devp).readlines()

    tt.msg("{} examples in training set".format(len(trainlines)))
    tt.msg("{} examples in dev set".format(len(devlines)))
    tt.msg("{} examples in test set".format(len(testlines)))

    nlsm = q.StringMatrix(freqcutoff=1)
    nlsm.tokenize = lambda x: x.strip().split()
    qlsm = q.StringMatrix(indicate_start_end=True, freqcutoff=1)
    qlsm.tokenize = lambda x: x.strip().split()

    i = 0
    for line in trainlines:
        nl, ql = line.split("\t")
        nlsm.add(nl)
        qlsm.add(ql)
        i += 1

    nlsm.unseen_mode = True
    qlsm.unseen_mode = True

    devstart = i

    for line in devlines:
        nl, ql = line.split("\t")
        nlsm.add(nl)
        qlsm.add(ql)
        i += 1

    teststart = i

    for line in testlines:
        nl, ql = line.split("\t")
        nlsm.add(nl)
        qlsm.add(ql)

    nlsm.finalize()
    qlsm.finalize()
    tt.tock("data loaded")

    return nlsm, qlsm, (devstart, teststart)


def run(lr=0.001,
        dropout=0.2,
        batsize=50,
        embdim=50,
        encdim=50,
        decdim=50,
        numlayers=1,
        bidir=False,
        which="geo",        # "geo", "atis", "jobs"
        test=True,
        ):
    settings = locals().copy()
    logger = q.log.Logger(prefix="seq2seq_base")
    logger.save_settings(**settings)
    # region data
    nlsm, qlsm, splitidxs = load_data(which=which)
    print(nlsm[0], qlsm[0])
    print(nlsm._rarewords)

    trainloader = q.dataload(nlsm.matrix[:splitidxs[0]], qlsm.matrix[:splitidxs[0]], batch_size=batsize, shuffle=True)
    devloader = q.dataload(nlsm.matrix[splitidxs[0]:splitidxs[1]], qlsm.matrix[splitidxs[0]:splitidxs[1]], batch_size=batsize, shuffle=False)
    testloader = q.dataload(nlsm.matrix[splitidxs[1]:], qlsm.matrix[splitidxs[1]:], batch_size=batsize, shuffle=False)
    # endregion

    # region model
    encdims = [encdim] * numlayers
    outdim = (encdim if not bidir else encdim * 2) + decdim
    nlemb = q.WordEmb(embdim, worddic=nlsm.D)
    qlemb = q.WordEmb(embdim, worddic=qlsm.D)
    nlenc = q.LSTMEncoder(embdim, *encdims, bidir=bidir, dropout_in=dropout)
    att = q.att.DotAtt()
    if numlayers > 1:
        qldec_core = torch.nn.Sequential(
            *[q.LSTMCell(_indim, _outdim, dropout_in=dropout)
              for _indim, _outdim in [(embdim, decdim)] + [(decdim, decdim)] * (numlayers - 1)]
        )
    else:
        qldec_core = q.LSTMCell(embdim, decdim, dropout_in=dropout)
    qlout = q.WordLinout(outdim, worddic=qlsm.D)
    qldec = q.LuongCell(emb=qlemb, core=qldec_core, att=att, out=qlout)

    class Model(torch.nn.Module):
        def __init__(self, _nlemb, _nlenc, _qldec, train=True, **kw):
            super(Model, self).__init__(**kw)
            self.nlemb, self.nlenc, self._q_train = _nlemb, _nlenc, train
            if train:
                self.qldec = q.TFDecoder(_qldec)
            else:
                self.qldec = q.FreeDecoder(_qldec, maxtime=100)

        def forward(self, x, y):   # (batsize, seqlen) int ids
            xemb, xmask = self.nlemb(x)
            xenc = self.nlenc(xemb, mask=xmask)
            if self._q_train is False:
                assert(y.dim() == 2)
            dec = self.qldec(y, ctx=xenc, ctxmask=xmask[:, :xenc.size(1)])
            return dec

    m_train = Model(nlemb, nlenc, qldec, train=True)
    m_test = Model(nlemb, nlenc, qldec, train=False)

    if test:
        test_out = m_train(torch.tensor(nlsm.matrix[:5]), torch.tensor(qlsm.matrix[:5]))
        print("test_out.size() = {}".format(test_out.size()))
    # endregion




if __name__ == '__main__':
    q.argprun(run)