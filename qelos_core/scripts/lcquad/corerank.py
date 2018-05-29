import qelos_core as q
import torch
import numpy as np
import json
import pickle
import random
import os


OPT_LR = 0.001


# TODO: DO RARE after loading


def load_jsons(datap="../../../datasets/lcquad/id_big_data.json",
               relp="../../../datasets/lcquad/nrels.json",
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
            ret = " ".join(flatchainspec).lower()
            return ret

        goldchainids = []
        badchainsids = []

        uniquechainids = {}

        qsm = q.StringMatrix()
        csm = q.StringMatrix()
        csm.tokenize = lambda x: x.strip().split()

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
            firstrel = u"" + chainspec[0] + u" " + u" ".join(rels[str(chainspec[1])])
            firstrel = firstrel.lower()
            secondrel = u"EMPTYEMPTYEMPTY"
            if len(chainspec) > 2 and chainspec[2] != -1:
                secondrel = u"" + chainspec[2] + u" " + u" ".join(rels[str(chainspec[3])])
                secondrel = secondrel.lower()
            return firstrel, secondrel

        goldchainids = []
        badchainsids = []

        uniquechainids = {}

        qsm = q.StringMatrix()
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
            qsm.add(question + u" EMPTYEMPTYEMPTY")
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
                if i < self.k:
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


class BestWriter(object):
    def __init__(self, qsm, csm, p=None, **kw):
        super(BestWriter, self).__init__()
        self.qsm = qsm
        self.csm = csm
        self.p = p

    def compute(self, rankings, **kw):
        ds = []
        with open(self.p, "w") as f:
            for eid, ranking in rankings:
                question_of_example = self.qsm[eid]
                best_scored_chain_of_example = self.csm[ranking[0][1]]
                est_truth_of_best_scored_chain = str(bool(ranking[0][2])).lower()
                number_of_chains_in_ranking_for_example = len(ranking)
                d = {"eid": eid,
                     "question": question_of_example,
                     "best_chain": best_scored_chain_of_example,
                     "best_chain_ass_truth": est_truth_of_best_scored_chain,
                     "num_chains": number_of_chains_in_ranking_for_example}
                ds.append(d)
            json.dump(ds, f, indent=2, sort_keys=True)
        return 0


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

        return self.compute_loss(psim, nsim)

    def compute_loss(self, psim, nsim):
        diffs = psim - nsim
        zeros = q.var(torch.zeros_like(diffs.data)).cuda(diffs).v
        losses = torch.max(zeros, self.margin - diffs)
        return losses


class RankModelPointwise(RankModel):
    def compute_loss(self, psim, nsim):
        ploss = -torch.log(torch.nn.Sigmoid(psim))
        nloss = -torch.log(1 - torch.nn.Sigmoid(nsim))
        interp = (torch.randn(ploss.size()) > 0).float()
        loss = ploss * interp + nloss * (1 - interp)
        return loss


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
    def __init__(self, embdim, dims, word_dic, bidir=False, dropout_in=0., dropout_rec=0., gfrac=0., meanpoolskip=False):
        """ embdim for embedder, dims is a list of dims for RNN"""
        super(FlatEncoder, self).__init__()
        self.emb = q.PartiallyPretrainedWordEmb(embdim, worddic=word_dic, gradfracs=(1., gfrac))
        self.lstm = q.FastestLSTMEncoder(embdim, *dims, bidir=bidir, dropout_in=dropout_in, dropout_rec=dropout_rec)
        self.meanpoolskip = meanpoolskip
        self.adapt_lin = None
        outdim = dims[-1] * 2
        if meanpoolskip and outdim != embdim:
            self.adapt_lin = torch.nn.Linear(embdim, outdim, bias=False)

    def forward(self, x):
        embs, mask = self.emb(x)
        _ = self.lstm(embs, mask=mask)
        final_state = self.lstm.y_n[-1]
        final_state = final_state.contiguous().view(x.size(0), -1)
        if self.meanpoolskip:
            if self.adapt_lin is not None:
                embs = self.adapt_lin(embs)
            meanpool = embs.sum(1)
            masksum = mask.float().sum(1).unsqueeze(1)
            meanpool = meanpool / masksum
            final_state = final_state + meanpool
        return final_state


def get_seen_words(qmat, qsmD, rarefreq=1, gdic=None):
    rD = {v: k for k, v in qsmD.items()}
    # get words in pretrained:
    gwords = set(gdic.keys())
    # rare words in qmat:
    uniquewordids, wordid_counts = np.unique(qmat, return_counts=True)
    wordid_isseen = wordid_counts >= rarefreq
    unique_seen_wordids = uniquewordids * (wordid_isseen).astype("int32")
    unique_seen_wordids = set(unique_seen_wordids)
    unique_seen_words = set([rD[unrid] for unrid in unique_seen_wordids])
    unique_seen_words |= gwords
    all_words = set(qsmD.keys())
    rare_words = all_words - unique_seen_words
    print("{} ({:.2f}%) rare words (from {} total), {} seen words not in g".format(len(rare_words), len(rare_words) * 100. / len(all_words), len(all_words), len(unique_seen_words - gwords)))
    return unique_seen_words


def get_seen_words_chains(eids, csm, goldchains, rarefreq=1, gdic=None):
    # get seen words from gold chains
    goldchains = np.asarray(goldchains)
    seen_goldchains = goldchains[eids]
    seen_chains = csm.matrix[seen_goldchains]
    unique_seen_words = get_seen_words(seen_chains, csm.D, rarefreq=rarefreq, gdic=gdic)
    return unique_seen_words


def replace_rare(mat, words, D):
    ids = set([D[word] for word in words if word in D])
    outmat = np.vectorize(lambda x: D["<RARE>"] if x not in ids else x)(mat)
    return outmat


def run(lr=OPT_LR, batsize=100, epochs=1000, validinter=20,
        wreg=0.00000000001, dropout=0.1,
        embdim=50, encdim=50, numlayers=1,
        cuda=False, gpu=0, mode="flat",
        test=False, gendata=False,
        seenfreq=0, beta2=0.999,
        validontest=False,
        pointwise=False):
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
        if not validontest:
            traindata, validdata = q.datasplit(data, splits=(7, 3), random=False)
            validdata, testdata = q.datasplit(validdata, splits=(1, 2), random=False)
        else:
            traindata, validdata = q.datasplit(data, splits=(8, 2), random=False)
            testdata = validdata

        if seenfreq > 0:
            gdic = q.PretrainedWordEmb(embdim).D
            seen_words = get_seen_words(traindata[0], qsm.D, rarefreq=seenfreq, gdic=gdic)
            seen_words_chains = get_seen_words_chains(traindata[1], csm, goldchainids, rarefreq=seenfreq, gdic=gdic)
            traindata[0], validdata[0], testdata[0] = [replace_rare(x, seen_words, qsm.D) for x in [traindata[0], validdata[0], testdata[0]]]
            csm._matrix = replace_rare(csm.matrix, seen_words_chains, csm.D)
            tt.msg("replaced rare words")

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

        rankmc = RankModel if not pointwise else RankModelPointwise
        rankmodel = rankmc(question_encoder, query_encoder, similarity)
        scoremodel = ScoreModel(question_encoder, query_encoder, similarity)
        # endregion

        # region TRAINING
        optim = torch.optim.Adam(q.params_of(rankmodel), lr=lr, betas=(0.9, beta2), weight_decay=wreg)
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
                                               MRR(),
                                               BestWriter(qsm, csm, os.path.join(logger.p, "valid.out.temp")))
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

        class WritingValidator(object):
            def __init__(self, _rankcomp, p=None):
                self.save_crit = -1.
                self.rankcomp = _rankcomp
                self.p = p

            def __call__(self):
                rankmetrics = self.rankcomp.compute(RecallAt(1, totaltrue=1),
                                               RecallAt(5, totaltrue=1),
                                               MRR(),
                                               BestWriter(qsm, csm, self.p))
                ret = []
                for rankmetric in rankmetrics:
                    rankmetric = np.asarray(rankmetric)
                    # print(rankmetric.shape)
                    ret_i = rankmetric.mean()
                    ret.append(ret_i)
                self.save_crit = ret[0]     # saves criterium for best saving
                return " - ".join(map(lambda x: "{:.4f}".format(x), ret))

        train_validator = WritingValidator(rankcomp_train, p=os.path.join(logger.p, "train.out"))
        valid_validator = WritingValidator(rankcomp_valid, p=os.path.join(logger.p, "valid.out"))
        test_validator = WritingValidator(rankcomp_test, p=os.path.join(logger.p, "test.out"))
        tt.msg("computing train metrics")
        train_results = train_validator()
        tt.msg("train results: {}".format(train_results))
        tt.msg("computing valid metrics")
        valid_results = valid_validator()
        tt.msg("valid results: {}".format(valid_results))
        tt.msg("computing test results")
        test_results = test_validator()
        tt.msg("test results: {}".format(test_results))

        with open(os.path.join(logger.p, "results.txt"), "w") as f:
            f.write("train results: {}\n".format(train_results))
            f.write("valid results: {}\n".format(valid_results))
            f.write("test results: {}\n".format(test_results))

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
        outdim = dims[-1] * 2
        self.adapt_lin = None
        if outdim != embdim:
            self.adapt_lin = torch.nn.Linear(embdim, outdim, bias=False)

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
        # region skipper
        skipadd = embs[:, :ys.size(1), :]
        if self.adapt_lin is not None:
            skipadd = self.adapt_lin(skipadd)
        nys = ys + skipadd
        # endregion
        nys = nys.unsqueeze(2)      # (batsize, seqlen, 1, dim)
        scores = scores.unsqueeze(3)    # (batsize, seqlen, 2, 1)
        b = nys * scores                # (batsize, seqlen, 2, dim)
        summaries = b.sum(1)        # (batsize, 2, dim)
        ret = torch.cat([summaries[:, 0, :], summaries[:, 1, :]], 1)
        return ret


class SlotPtrChainEncoder(torch.nn.Module):
    def __init__(self, embdim, dims, word_dic, firstrellen, bidir=False, dropout_in=0., dropout_rec=0., gfrac=0., meanpoolskip=False):
        super(SlotPtrChainEncoder, self).__init__()
        self.firstrellen = firstrellen
        self.enc = FlatEncoder(embdim, dims, word_dic, bidir=bidir, dropout_in=dropout_in,
                               dropout_rec=dropout_rec, gfrac=gfrac, meanpoolskip=meanpoolskip)

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
        test=False, gendata=False,
        seenfreq=0, beta2=0.999,
        meanpoolskip=True,
        validontest=False,
        pointwise=False):
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
        if not validontest:
            traindata, validdata = q.datasplit(data, splits=(7, 3), random=False)
            validdata, testdata = q.datasplit(validdata, splits=(1, 2), random=False)
        else:
            traindata, validdata = q.datasplit(data, splits=(8, 2), random=False)
            testdata = validdata

        if seenfreq > 0:
            gdic = q.PretrainedWordEmb(embdim).D
            seen_words = get_seen_words(traindata[0], qsm.D, rarefreq=seenfreq, gdic=gdic)
            seen_words_chains = get_seen_words_chains(traindata[1], csm, goldchainids, rarefreq=seenfreq, gdic=gdic)
            traindata[0], validdata[0], testdata[0] = [replace_rare(x, seen_words, qsm.D) for x in [traindata[0], validdata[0], testdata[0]]]
            csm._matrix = replace_rare(csm.matrix, seen_words_chains, csm.D)
            tt.msg("replaced rare words")

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
        query_encoder = SlotPtrChainEncoder(embdim, dims, csm.D, maxfirstrellen, bidir=True, dropout_in=dropout, dropout_rec=dropout,
                                            meanpoolskip=meanpoolskip)
        similarity = DotDistance()

        rankmc = RankModel if not pointwise else RankModelPointwise
        rankmodel = rankmc(question_encoder, query_encoder, similarity)
        scoremodel = ScoreModel(question_encoder, query_encoder, similarity)
        # endregion

        # region TRAINING
        optim = torch.optim.Adam(q.params_of(rankmodel), lr=lr, betas=(0.9, beta2), weight_decay=wreg)
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
                                                    MRR(),
                                                    BestWriter(qsm, csm, os.path.join(logger.p, "valid.out.temp")))
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

        class WritingValidator(object):
            def __init__(self, _rankcomp, p=None):
                self.save_crit = -1.
                self.rankcomp = _rankcomp
                self.p = p

            def __call__(self):
                rankmetrics = self.rankcomp.compute(RecallAt(1, totaltrue=1),
                                               RecallAt(5, totaltrue=1),
                                               MRR(),
                                               BestWriter(qsm, csm, self.p))
                ret = []
                for rankmetric in rankmetrics:
                    rankmetric = np.asarray(rankmetric)
                    # print(rankmetric.shape)
                    ret_i = rankmetric.mean()
                    ret.append(ret_i)
                self.save_crit = ret[0]     # saves criterium for best saving
                return " - ".join(map(lambda x: "{:.4f}".format(x), ret))

        train_validator = WritingValidator(rankcomp_train, p=os.path.join(logger.p, "train.out"))
        valid_validator = WritingValidator(rankcomp_valid, p=os.path.join(logger.p, "valid.out"))
        test_validator = WritingValidator(rankcomp_test, p=os.path.join(logger.p, "test.out"))
        tt.msg("computing train metrics")
        train_results = train_validator()
        tt.msg("train results: {}".format(train_results))

        # region OLD --> remove
        # train_validator = Validator(rankcomp_train)
        # valid_validator = Validator(rankcomp_valid)
        # test_validator = Validator(rankcomp_test)
        # tt.msg("computing train metrics")
        # train_results = train_validator()
        # tt.msg("train results: {}".format(train_results))
        # endregion
        tt.msg("computing valid metrics")
        valid_results = valid_validator()
        tt.msg("valid results: {}".format(valid_results))
        tt.msg("computing test results")
        test_results = test_validator()
        tt.msg("test results: {}".format(test_results))

        with open(os.path.join(logger.p, "results.txt"), "w") as f:
            f.write("train results: {}\n".format(train_results))
            f.write("valid results: {}\n".format(valid_results))
            f.write("test results: {}\n".format(test_results))

        tt.tock("computed metrics")
        # endregion


if __name__ == "__main__":
    # q.argprun(run)
    q.argprun(run_slotptr)