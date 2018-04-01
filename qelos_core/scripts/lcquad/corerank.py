import qelos_core as q
import torch
import numpy as np
import json
import pickle
import random


OPT_LR = 0.001


# TODO: port metrics


def load_jsons(datap="../../../datasets/lcquad/newdata.json",
               relp="../../../datasets/lcquad/nrelations.json",
               mode="flat"):
    tt = q.ticktock("data loader")
    tt.tick("loading jsons")

    data = json.load(open(datap))
    rels = json.load(open(relp))

    tt.tock("jsons loaded")

    tt.tick("extracting data")
    questions = []
    goldchains = []
    badchains = []
    for dataitem in data:
        questions.append(dataitem["parsed-data"]["corrected_question"])
        goldchain = []
        for x in dataitem["parsed-data"]["path_id"]:
            goldchain += [x[0], int(x[1:])]
        goldchains.append(goldchain)
        badchainses = []
        goldfound = False
        for badchain in dataitem["uri"]["hop-1-properties"] + dataitem["uri"]["hop-2-properties"]:
            if goldchain == badchain:
                goldfound = True
            else:
                if len(badchain) == 2:
                    badchain += [-1, -1]
                badchainses.append(badchain)
        badchains.append(badchainses)

    tt.tock("extracted data")

    tt.msg("mode: {}".format(mode))

    if mode == "flat":
        tt.tick("flattening")

        def flatten_chain(chainspec):
            flatchainspec = []
            for x in chainspec:
                if x in (u"+", u"-"):
                    flatchainspec.append(x)
                elif x > -1:
                    relwords = rels[str(x)]
                    flatchainspec += relwords
                elif x == -1:
                    pass
                else:
                    raise q.SumTingWongException("unexpected symbol in chain")
            return " ".join(flatchainspec)

        goldchainids = []
        badchainsids = []

        uniquechainids = {}

        qsm = q.StringMatrix()
        csm = q.StringMatrix()
        csm.tokenize = lambda x: x.lower().strip().split()

        def get_ensure_chainid(flatchain):
            if flatchain not in uniquechainids:
                uniquechainids[flatchain] = len(uniquechainids)
                csm.add(flatchain)
                assert(len(csm) == len(uniquechainids))
            return uniquechainids[flatchain]

        eid = 0
        numchains = 0
        for question, goldchain, badchainses in zip(questions, goldchains, badchains):
            qsm.add(question)
            # flatten gold chain
            flatgoldchain = flatten_chain(goldchain)
            chainid = get_ensure_chainid(flatgoldchain)
            goldchainids.append(chainid)
            badchainsids.append([])
            numchains += 1
            for badchain in badchainses:
                flatbadchain = flatten_chain(badchain)
                chainid = get_ensure_chainid(flatbadchain)
                badchainsids[eid].append(chainid)
                numchains += 1
            eid += 1
            tt.live("{}".format(eid))

        assert(len(badchainsids) == len(questions))
        tt.stoplive()
        tt.msg("{} unique chains from {} total".format(len(csm), numchains))
        qsm.finalize()
        csm.finalize()
        tt.tock("flattened")
        csm.tokenize = None
        return qsm, csm, goldchainids, badchainsids
    else:
        raise q.SumTingWongException("unsupported mode: {}".format(mode))


class RankingComputer(object):
    """ computes rankings based on ranking model for full validation/test ranking
        provides separate loss objects to put into lossarray
    """

    def __init__(self, scoremodel, eids, ldata, rdata, eid2rid_gold, eid2rid_neg):
        self.scoremodel = scoremodel
        self.eids = eids
        self.ldata = ldata if q.issequence(ldata) else (ldata,)     # already shuffled
        self.rdata = rdata if q.issequence(rdata) else (rdata,)     # indexed by eid space
        self.eid2rid_neg = eid2rid_neg      # indexed by eid space
        self.eid2rid_gold = eid2rid_gold    # indexed by eid space

    def compute(self, *metrics):        # compute given metrics for all given data
        self.scoremodel.train(False)
        rankings = self.compute_rankings(self.eids)
        metricnumbers = []
        for i, metric in enumerate(metrics):
            metricnumbers.append(metric.compute(rankings))
        # TODO
        return metricnumbers

    def compute_rankings(self, eids):
        cuda = q.iscuda(self.scoremodel)
        # get all pairs to score
        current_batch = []
        # given questions are already shuffled --> just traverse
        for eid, ldata_id in zip(list(eids), range(len(self.eids))):
            rdata = []
            rids = [self.eid2rid_gold[eid]] + list(set(self.eid2rid_neg[eid]) - {self.eid2rid_gold[eid],})
            ldata = [ldat[ldata_id][np.newaxis, ...].repeat(len(rids), axis=0)
                          for ldat in self.ldata]
            trueornot = [0] * len(rids)
            trueornot[0] = 1
            for rid in rids:
                right_data = tuple([rdat[rid] for rdat in self.rdata])
                rdata.append(right_data)
            rdata = zip(*rdata)
            ldata = [q.var(ldat, volatile=True).cuda(cuda).v for ldat in ldata]
            rdata = [q.var(np.stack(posdata_e), volatile=True).cuda(cuda).v for posdata_e in rdata]
            scores = self.scoremodel(ldata, rdata)
            _scores = list(scores.cpu().data.numpy())
            ranking = sorted(zip(_scores, rids, trueornot), key=lambda x: x[0], reverse=True)
            current_batch.append((eid, ranking))
        return current_batch


class RecallAt(object):
    def __init__(self, k, totaltrue=None, **kw):
        super(RecallAt, self).__init__(**kw)
        self.k = k
        self.totaltrue = totaltrue

    def compute(self, rankings, **kw):
        # list or (eid, lid, ranking)
        # ranking is a list of triples (_scores, rids, trueornot)
        ys = []
        for _, ranking in rankings:
            topktrue = 0.
            totaltrue = 0.
            for i in range(len(ranking)):
                _, _, trueornot = ranking[i]
                if i <= self.k:
                    topktrue += trueornot
                else:
                    if self.totaltrue is not None:
                        totaltrue = self.totaltrue
                        break
                if trueornot == 1:
                    totaltrue += 1.
            topktrue = topktrue / totaltrue
            ys.append(topktrue)
        ys = np.stack(ys)
        return ys


class MRR(object):
    def compute(self, rankings, **kw):
        # list or (eid, lid, ranking)
        # ranking is a list of triples (_scores, rids, trueornot)
        ys = []
        for _, ranking in rankings:
            res = 0
            for i in range(len(ranking)):
                _, _, trueornot = ranking[i]
                if trueornot == 1:
                    res = 1./(i+1)
                    break
            ys.append(res)
        ys = np.stack(ys)
        return ys


class FlatInpFeeder(object):
    """ samples RHS data only """
    def __init__(self, csm, goldcids, badcids):
        """ goldcids: 1D list, badcids: list of lists"""
        super(FlatInpFeeder, self).__init__()
        self.csm = csm
        self.goldcids = goldcids
        self.badcids = badcids

    def __call__(self, eids):
        # golds is (batsize, seqlen)
        golds = np.zeros((len(eids), self.csm.shape[1]), dtype="int64")
        bads = np.zeros((len(eids), self.csm.shape[1]), dtype="int64")

        for i, eid in enumerate(eids.cpu().data.numpy()):
            golds[i, :] = self.csm[self.goldcids[eid]]
            badcidses = self.badcids[eid]
            if len(badcidses) == 0:
                badcid = random.randint(0, len(self.csm))
            else:
                badcid = random.sample(badcidses, 1)[0]
            bads[i, :] = self.csm[badcid]

        golds = q.var(golds).cuda(eids).v
        bads = q.var(bads).cuda(eids).v
        return golds, bads


class RankModel(torch.nn.Module):
    """ Does the margin, and the loss --> use with LinearLoss """
    def __init__(self, lmodel, rmodel, similarity, margin=1., **kw):
        super(RankModel, self).__init__(**kw)
        self.lmodel = lmodel
        self.rmodel = rmodel
        self.sim = similarity
        self.margin = margin

    def forward(self, ldata, posrdata, negrdata):
        ldata = ldata if q.issequence(ldata) else (ldata,)
        posrdata = posrdata if q.issequence(posrdata) else (posrdata,)
        negrdata = negrdata if q.issequence(negrdata) else (negrdata,)
        lvecs = self.lmodel(*ldata)      # 2D
        rvecs = self.rmodel(*posrdata)      # 2D
        nrvecs = self.rmodel(*negrdata)
        psim = self.sim(lvecs, rvecs)    # 1D:(batsize,)
        nsim = self.sim(lvecs, nrvecs)

        diffs = psim - nsim
        zeros = q.var(torch.zeros_like(diffs.data)).cuda(diffs).v
        losses = torch.max(zeros, self.margin - diffs)

        return losses


class ScoreModel(torch.nn.Module):
    def __init__(self, lmodel, rmodel, similarity, **kw):
        super(ScoreModel, self).__init__(**kw)
        self.lmodel = lmodel
        self.rmodel = rmodel
        self.sim = similarity

    def forward(self, ldata, rdata):
        ldata = ldata if q.issequence(ldata) else (ldata,)
        rdata = rdata if q.issequence(rdata) else (rdata,)
        # q.embed()
        lvecs = self.lmodel(*ldata)      # 2D
        rvecs = self.rmodel(*rdata)      # 2D
        psim = self.sim(lvecs, rvecs)    # 1D:(batsize,)
        return psim


class Distance(torch.nn.Module):
    pass


class DotDistance(Distance):        # actually a similarity
    def forward(self, data, crit):        # (batsize, seqlen, dim), (batsize, dim)
        datadim, critdim = data.dim(), crit.dim()
        if data.dim() == 2:               # if datasets is (batsize, dim),
            data = data.unsqueeze(1)      #     make (batsize, 1, dim)
        if crit.dim() == 2:               # if crit is (batsize, dim)
            crit = crit.unsqueeze(-1)     #     make crit (batsize, dim, 1)
        else:                             # else crit must be (batsize, seqlen, dim)
            crit = crit.permute(0, 2, 1)  #     but we need (batsize, dim, seqlen)
        dist = torch.bmm(data, crit)      # batched mat dot --> (batsize,1,1) or (batsize, lseqlen,1) or (batsize, lseqlen, rseqlen)
        ret = dist.squeeze(1) if datadim == 2 else dist
        ret = ret.squeeze(-1) if critdim == 2 else ret
        return ret


class FlatEncoder(torch.nn.Module):
    def __init__(self, embdim, dims, word_dic, bidir=False, dropout_in=0., dropout_rec=0., gfrac=0.):
        """ embdim for embedder, dims is a list of dims for RNN"""
        super(FlatEncoder, self).__init__()
        self.emb = q.PartiallyPretrainedWordEmb(embdim, worddic=word_dic, gradfracs=(1., gfrac))
        self.lstm = q.FastestLSTMEncoder(embdim, *dims, bidir=bidir, dropout_in=dropout_in, dropout_rec=dropout_rec)

    def forward(self, x):
        embs, mask = self.emb(x)
        _ = self.lstm(embs, mask=mask)
        final_state = self.lstm.y_n[-1]
        final_state = final_state.contiguous().view(x.size(0), -1)
        return final_state


def run(lr=OPT_LR, batsize=100, epochs=100, validinter=5,
        wreg=0.00000000001, dropout=0.1,
        embdim=50, encdim=50, numlayers=1,
        cuda=False, gpu=0, mode="flat",
        test=False):
    settings = locals().copy()
    logger = q.Logger(prefix="rank_lstm")
    logger.save_settings(**settings)
    if cuda:
        torch.cuda.set_device(gpu)

    tt = q.ticktock("script")

    # region DATA
    tt.tick("loading data")
    qsm, csm, goldchainids, badchainids = pickle.load(open("loadcache.{}.pkl".format(mode)))
    eids = np.arange(0, len(goldchainids))

    data = [qsm.matrix, eids]
    traindata, validdata = q.datasplit(data, splits=(7, 3), random=False)
    validdata, testdata = q.datasplit(validdata, splits=(1, 2), random=False)

    trainloader = q.dataload(*traindata, batch_size=batsize, shuffle=True)

    input_feeder = FlatInpFeeder(csm.matrix, goldchainids, badchainids)

    def inp_bt(_qsm_batch, _eids_batch):
        golds_batch, bads_batch = input_feeder(_eids_batch)
        dummygold = _eids_batch
        return _qsm_batch, golds_batch, bads_batch, dummygold

    if test:
        # test input feeder
        eids = q.var(torch.arange(0, 10).long()).v
        _test_golds_batch, _test_bads_batch = input_feeder(eids)
    tt.tock("data loaded")
    # endregion

    # region MODEL
    dims = [encdim//2] * numlayers

    question_encoder = FlatEncoder(embdim, dims, qsm.D, bidir=True)
    query_encoder = FlatEncoder(embdim, dims, csm.D, bidir=True)
    similarity = DotDistance()

    rankmodel = RankModel(question_encoder, query_encoder, similarity)
    scoremodel = ScoreModel(question_encoder, query_encoder, similarity)
    # endregion

    # region VALIDATION
    rankcomp = RankingComputer(scoremodel, validdata[1], validdata[0],
                               csm.matrix, goldchainids, badchainids)
    # endregion

    # region TRAINING
    optim = torch.optim.Adam(q.params_of(rankmodel), lr=lr, weight_decay=wreg)
    trainer = q.trainer(rankmodel).on(trainloader).loss(q.LinearLoss())\
               .set_batch_transformer(inp_bt).optimizer(optim).cuda(cuda)

    def validation_function():
        rankmetrics = rankcomp.compute(RecallAt(1, totaltrue=1),
                                       RecallAt(5, totaltrue=1),
                                       MRR())
        ret = []
        for rankmetric in rankmetrics:
            rankmetric = np.asarray(rankmetric)
            ret_i = rankmetric.mean()
            ret.append(ret_i)
        return "valid: " + " - ".join(map(lambda x: "{:.4f}".format(x), ret))

    q.train(trainer, validation_function).run(epochs, validinter=validinter)
    # endregion


if __name__ == "__main__":
    # loadret = load_jsons()
    # pickle.dump(loadret, open("loadcache.flat.pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)
    q.argprun(run)