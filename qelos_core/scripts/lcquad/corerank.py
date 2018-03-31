import qelos_core as q
import torch
import numpy as np
import json
from qelos import StringMatrix
import pickle


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
        eid = 0
        sm_id = 0

        goldchainids = []
        badchainsids = []

        sm = StringMatrix()

        for question in questions:
            sm.add(question)
            sm_id += 1

        chainsstart = sm_id

        sm.tokenize = lambda x: x.lower().strip().split()

        for question, goldchain, badchainses in zip(questions, goldchains, badchains):
            # qsm.add(question)
            # flatten gold chain
            flatgoldchain = flatten_chain(goldchain)
            flatbadchainses = [flatten_chain(badchain) for badchain in badchainses]
            sm.add(flatgoldchain)
            goldchainids.append(sm_id)
            sm_id += 1
            badchainsids.append([])
            for flatbadchain in flatbadchainses:
                sm.add(flatbadchain)
                badchainsids[eid].append(sm_id)
                sm_id += 1
            eid += 1
            tt.live("{}".format(eid))

        assert(len(badchainsids) == len(questions))
        tt.stoplive()
        sm.finalize()
        tt.tock("flattened")
        return sm, chainsstart, goldchainids, badchainsids
    else:
        raise q.SumTingWongException("unsupported mode: {}".format(mode))


class QuestionEncoder(torch.nn.Module):
    def __init__(self, embdim, dims, word_dic, bidir=False, dropout_in=0., dropout_rec=0., gfrac=0.):
        """ embdim for embedder, dims is a list of dims for RNN"""
        super(QuestionEncoder, self).__init__()
        self.emb = q.PartiallyPretrainedWordEmb(embdim, worddic=word_dic, gradfracs=(1., gfrac))
        self.lstm = q.FastestLSTMEncoder(embdim, *dims, bidir=bidir, dropout_in=dropout_in, dropout_rec=dropout_rec)

    def forward(self, x):
        embs, mask = self.emb(x)
        _ = self.lstm(embs, mask=mask)
        final_state = self.lstm.y_n[-1]
        return final_state


def run(lr=OPT_LR, cuda=False, gpu=0):
    settings = locals().copy()
    logger = q.Logger(prefix="rank_lstm")
    logger.save_settings(**settings)
    if cuda:
        torch.cuda.set_device(gpu)

    tt = q.ticktock("script")
    tt.tick("loading data")
    # TODO: load data
    tt.tock("data loaded")

    question_encoder = QuestionEncoder(embdim, dims, word_dic)


if __name__ == "__main__":
    loadret = load_jsons()
    pickle.dump(loadret, open("loadcache.flat.pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)
    q.argprun(run)