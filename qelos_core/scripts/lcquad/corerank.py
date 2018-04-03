import qelos_core as q
import torch
import numpy as np
import json
import pickle
import random
import os


OPT_LR = 0.001


# TODO: DO RARE after loading


def load_jsons(datap="../../../datasets/lcquad/newdata.json",
               relp="../../../datasets/lcquad/nrelations.json",
               mode="flat"):
    """ relp: file must contain dictionary mapping relation ids (ints) to lists of words (strings)"""
    """ mode: "flat", "slotptr" """
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
    elif mode == "slotptr":
        """ adds a "<EMPTY>" at the end of the question,
            chain is 2D, with "<EMPTY>" at second relation if no second relation"""
        tt.tick("flattening")

        def flatten_chain(chainspec):
            flatchainspec = []
            firstrel = u"" + chainspec[0] + u" " + u" ".join(rels[str(chainspec[1])])
            secondrel = u"<EMPTY>"
            if len(chainspec) > 2 and chainspec[2] != -1:
                secondrel = u"" + chainspec[2] + u" " + u" ".join(rels[str(chainspec[3])])
            return firstrel, secondrel

        goldchainids = []
        badchainsids = []

        uniquechainids = {}

        qsm = q.StringMatrix()
        qsm.protectedwords = qsm.protectedwords + [u"<EMPTY>"]
        csm = q.StringMatrix()

        csm.tokenize = lambda x: x.strip().split()

        firstrels = []
        maxfirstrellen = [0]
        secondrels = []

        def get_ensure_chainid(flatchain1, flatchain2):
            flatchain = flatchain1 + u" " + flatchain2
            if flatchain not in uniquechainids:
                uniquechainids[flatchain] = len(uniquechainids)
                firstrels.append(flatchain1)
                secondrels.append(flatchain2)
                assert(len(firstrels) == len(uniquechainids))
                maxfirstrellen[0] = max(maxfirstrellen[0], len(flatchain1.strip().split()))
            return uniquechainids[flatchain]

        eid = 0
        numchains = 0
        for question, goldchain, badchainses in zip(questions, goldchains, badchains):
            qsm.add(question + u" <EMPTY>")
            # flatten gold chain
            flatgoldchain1, flatgoldchain2 = flatten_chain(goldchain)
            chainid = get_ensure_chainid(flatgoldchain1, flatgoldchain2)
            goldchainids.append(chainid)
            badchainsids.append([])
            numchains += 1
            for badchain in badchainses:
                flatbadchain1, flatbadchain2 = flatten_chain(badchain)
                chainid = get_ensure_chainid(flatbadchain1, flatbadchain2)
                badchainsids[eid].append(chainid)
                numchains += 1
            eid += 1
            tt.live("{}".format(eid))

        assert(len(badchainsids) == len(questions))
        tt.stoplive()
        tt.msg("{} unique chains from {} total".format(len(firstrels), numchains))
        qsm.finalize()
        # finalize csm
        for firstrel, secondrel in zip(firstrels, secondrels):
            firstrelsplits = firstrel.split()
            firstrelsplits = firstrelsplits + [u"<MASK>"] * (maxfirstrellen[0] - len(firstrelsplits))
            secondrelsplits = secondrel.split()
            secondrelsplits = secondrelsplits + [u"<MASK>"] * (maxfirstrellen[0] - len(secondrelsplits))
            rel = u" ".join(firstrelsplits + secondrelsplits)
            csm.add(rel)
        csm.finalize()
        tt.tock("flattened")
        csm.tokenize = None
        return qsm, csm, maxfirstrellen[0], goldchainids, badchainsids
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
        golds = np.zeros((len(eids),) + self.csm.shape[1:], dtype="int64")
        bads = np.zeros((len(eids),) + self.csm.shape[1:], dtype="int64")

        for i, eid in enumerate(eids.cpu().data.numpy()):
            golds[i, ...] = self.csm[self.goldcids[eid]]
            badcidses = self.badcids[eid]
            if len(badcidses) == 0:
                badcid = random.randint(0, len(self.csm)-1)
            else:
                badcid = random.sample(badcidses, 1)[0]
            bads[i, ...] = self.csm[badcid]

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


def run(lr=OPT_LR, batsize=100, epochs=1000, validinter=20,
        wreg=0.00000000001, dropout=0.1,
        embdim=50, encdim=50, numlayers=1,
        cuda=False, gpu=0, mode="flat",
        test=False, gendata=False):
    if gendata:
        loadret = load_jsons()
        pickle.dump(loadret, open("loadcache.flat.pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        settings = locals().copy()
        logger = q.Logger(prefix="rank_lstm")
        logger.save_settings(**settings)
        if cuda:
            torch.cuda.set_device(gpu)

        tt = q.ticktock("script")

        # region DATA
        tt.tick("loading data")
        qsm, csm, goldchainids, badchainids = pickle.load(open("loadcache.flat.pkl"))
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

        question_encoder = FlatEncoder(embdim, dims, qsm.D, bidir=True, dropout_in=dropout, dropout_rec=dropout)
        query_encoder = FlatEncoder(embdim, dims, csm.D, bidir=True, dropout_in=dropout, dropout_rec=dropout)
        similarity = DotDistance()

        rankmodel = RankModel(question_encoder, query_encoder, similarity)
        scoremodel = ScoreModel(question_encoder, query_encoder, similarity)
        # endregion

        # region TRAINING
        optim = torch.optim.Adam(q.params_of(rankmodel), lr=lr, weight_decay=wreg)
        trainer = q.trainer(rankmodel).on(trainloader).loss(q.LinearLoss())\
                   .set_batch_transformer(inp_bt).optimizer(optim).cuda(cuda)

        rankcomp = RankingComputer(scoremodel, validdata[1], validdata[0],
                                   csm.matrix, goldchainids, badchainids)
        class Validator(object):
            def __init__(self, _rankcomp):
                self.save_crit = -1.
                self.rankcomp = _rankcomp

            def __call__(self):
                rankmetrics = self.rankcomp.compute(RecallAt(1, totaltrue=1),
                                               RecallAt(5, totaltrue=1),
                                               MRR())
                ret = []
                for rankmetric in rankmetrics:
                    rankmetric = np.asarray(rankmetric)
                    # print(rankmetric.shape)
                    ret_i = rankmetric.mean()
                    ret.append(ret_i)
                self.save_crit = ret[0]     # saves criterium for best saving
                return " - ".join(map(lambda x: "{:.4f}".format(x), ret))

        validator = Validator(rankcomp)

        bestsaver = q.BestSaver(lambda : validator.save_crit,
                                scoremodel, os.path.join(logger.p, "best.model"),
                                autoload=True, verbose=True)

        q.train(trainer, validator).hook(bestsaver)\
            .run(epochs, validinter=validinter, print_on_valid_only=True)

        logger.update_settings(completed=True,
                               final_valid_acc=bestsaver.best_criterion)

        tt.tick("computing metrics on all data")
        rankcomp_train = RankingComputer(scoremodel, traindata[1], traindata[0],
                                         csm.matrix, goldchainids, badchainids)
        rankcomp_valid = RankingComputer(scoremodel, validdata[1], validdata[0],
                                         csm.matrix, goldchainids, badchainids)
        rankcomp_test =  RankingComputer(scoremodel, testdata[1],  testdata[0],
                                         csm.matrix, goldchainids, badchainids)
        train_validator = Validator(rankcomp_train)
        valid_validator = Validator(rankcomp_valid)
        test_validator = Validator(rankcomp_test)
        tt.msg("computing train metrics")
        train_results = train_validator()
        tt.msg("train results: {}".format(train_results))
        tt.msg("computing valid metrics")
        valid_results = valid_validator()
        tt.msg("valid results: {}".format(valid_results))
        tt.msg("computing test results")
        test_results = test_validator()
        tt.msg("test results: {}".format(test_results))

        tt.tock("computed metrics")
        # endregion


class SlotPtrQuestionEncoder(torch.nn.Module):
    # TODO: (1) skip connection, (2) two outputs (summaries weighted by forwards)
    def __init__(self, embdim, dims, word_dic, bidir=False, dropout_in=0., dropout_rec=0., gfrac=0.):
        super(SlotPtrQuestionEncoder, self).__init__()
        self.emb = q.PartiallyPretrainedWordEmb(embdim, worddic=word_dic, gradfracs=(1., gfrac))
        self.lstm = q.FastestLSTMEncoder(embdim, *dims, bidir=bidir, dropout_in=dropout_in, dropout_rec=dropout_rec)
        self.linear = torch.nn.Linear(dims[-1]*2, 2)
        self.sm = torch.nn.Softmax(1)

    def forward(self, x):
        embs, mask = self.emb(x)
        ys = self.lstm(embs, mask=mask)
        final_state = self.lstm.y_n[-1]
        final_state = final_state.contiguous().view(x.size(0), -1)
        # get attention scores
        scores = self.linear(ys)
        scores = scores + torch.log(mask[:, :ys.size(1)].float().unsqueeze(2))
        scores = self.sm(scores)    # (batsize, seqlen, 2)
        # get summaries
        nys = ys + embs[:, :ys.size(1), :]     # skipper
        nys = nys.unsqueeze(2)      # (batsize, seqlen, 1, dim)
        scores = scores.unsqueeze(3)    # (batsize, seqlen, 2, 1)
        b = nys * scores                # (batsize, seqlen, 2, dim)
        summaries = b.sum(1)        # (batsize, 2, dim)
        ret = torch.cat([summaries[:, 0, :], summaries[:, 1, :]], 1)
        return ret


class SlotPtrChainEncoder(torch.nn.Module):
    def __init__(self, embdim, dims, word_dic, firstrellen, bidir=False, dropout_in=0., dropout_rec=0., gfrac=0.):
        super(SlotPtrChainEncoder, self).__init__()
        self.firstrellen = firstrellen
        self.enc = FlatEncoder(embdim, dims, word_dic, bidir=bidir, dropout_in=dropout_in, dropout_rec=dropout_rec, gfrac=gfrac)

    def forward(self, x):
        firstrels = x[:, :self.firstrellen]
        secondrels = x[:, self.firstrellen:]
        firstrels_enc = self.enc(firstrels)
        secondrels_enc = self.enc(secondrels)
        # cat???? # TODO
        enc = torch.cat([firstrels_enc, secondrels_enc], 1)
        return enc


def run_slotptr(lr=OPT_LR, batsize=100, epochs=1000, validinter=20,
        wreg=0.00000000001, dropout=0.1,
        embdim=50, encdim=50, numlayers=1,
        cuda=False, gpu=0,
        test=False, gendata=False):
    if gendata:
        loadret = load_jsons(mode="slotptr")
        pickle.dump(loadret, open("loadcache.slotptr.pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        settings = locals().copy()
        logger = q.Logger(prefix="slotptr")
        logger.save_settings(**settings)
        if cuda:
            torch.cuda.set_device(gpu)

        tt = q.ticktock("script")

        # region DATA
        tt.tick("loading data")
        qsm, csm, maxfirstrellen, goldchainids, badchainids = pickle.load(open("loadcache.slotptr.pkl"))
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

        question_encoder = SlotPtrQuestionEncoder(embdim, dims, qsm.D, bidir=True, dropout_in=dropout, dropout_rec=dropout)
        query_encoder = SlotPtrChainEncoder(embdim, dims, csm.D, maxfirstrellen, bidir=True, dropout_in=dropout, dropout_rec=dropout)
        similarity = DotDistance()

        rankmodel = RankModel(question_encoder, query_encoder, similarity)
        scoremodel = ScoreModel(question_encoder, query_encoder, similarity)
        # endregion

        # region TRAINING
        optim = torch.optim.Adam(q.params_of(rankmodel), lr=lr, weight_decay=wreg)
        trainer = q.trainer(rankmodel).on(trainloader).loss(q.LinearLoss())\
                   .set_batch_transformer(inp_bt).optimizer(optim).cuda(cuda)

        rankcomp = RankingComputer(scoremodel, validdata[1], validdata[0],
                                   csm.matrix, goldchainids, badchainids)

        class Validator(object):
            def __init__(self, _rankcomp):
                self.save_crit = -1.
                self.rankcomp = _rankcomp

            def __call__(self):
                rankmetrics = self.rankcomp.compute(RecallAt(1, totaltrue=1),
                                                    RecallAt(5, totaltrue=1),
                                                    MRR())
                ret = []
                for rankmetric in rankmetrics:
                    rankmetric = np.asarray(rankmetric)
                    # print(rankmetric.shape)
                    ret_i = rankmetric.mean()
                    ret.append(ret_i)
                self.save_crit = ret[0]  # saves criterium for best saving
                return " - ".join(map(lambda x: "{:.4f}".format(x), ret))

        validator = Validator(rankcomp)

        bestsaver = q.BestSaver(lambda: validator.save_crit,
                                scoremodel, os.path.join(logger.p, "best.model"),
                                autoload=True, verbose=True)

        q.train(trainer, validator).hook(bestsaver) \
            .run(epochs, validinter=validinter, print_on_valid_only=True)

        logger.update_settings(completed=True,
                               final_valid_acc=bestsaver.best_criterion)

        tt.tick("computing metrics on all data")
        rankcomp_train = RankingComputer(scoremodel, traindata[1], traindata[0],
                                         csm.matrix, goldchainids, badchainids)
        rankcomp_valid = RankingComputer(scoremodel, validdata[1], validdata[0],
                                         csm.matrix, goldchainids, badchainids)
        rankcomp_test = RankingComputer(scoremodel, testdata[1], testdata[0],
                                        csm.matrix, goldchainids, badchainids)
        train_validator = Validator(rankcomp_train)
        valid_validator = Validator(rankcomp_valid)
        test_validator = Validator(rankcomp_test)
        tt.msg("computing train metrics")
        train_results = train_validator()
        tt.msg("train results: {}".format(train_results))
        tt.msg("computing valid metrics")
        valid_results = valid_validator()
        tt.msg("valid results: {}".format(valid_results))
        tt.msg("computing test results")
        test_results = test_validator()
        tt.msg("test results: {}".format(test_results))

        tt.tock("computed metrics")
        # endregion


if __name__ == "__main__":
    # q.argprun(run)
    q.argprun(run_slotptr)