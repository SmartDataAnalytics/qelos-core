import json
import torch
import qelos_core as q
import numpy as np
import os
import unidecode
import re
import pickle
import csv


DATA_PREFIX = "../../../datasets/simpledbpediaqa/"


def tocsv(which="train"):
    inpp = os.path.join(DATA_PREFIX, which + ".json")
    outp = os.path.join(DATA_PREFIX, which + ".csv")
    oneoutp = os.path.join(DATA_PREFIX, which + "one.csv")
    inp = json.load(open(inpp, "r"))
    with open(outp, "w") as f, open(oneoutp, "w") as onef:
        fwriter = csv.writer(f, delimiter=",")
        onefwriter = csv.writer(onef, delimiter=",")
        fwriter.writerow(["question", "relation"])
        onefwriter.writerow(["question", "relation"])
        for question in inp["Questions"]:
            qtext = question["Query"]
            firstrel = None
            allrels = set()
            for pred in question["PredicateList"]:
                qrel = ("-" if pred["Direction"] == "backward" else "") + pred["Predicate"]
                if firstrel is None:
                    firstrel = qrel
                allrels.add(qrel)
            onefwriter.writerow([qtext, firstrel])
            for qrel in allrels:
                fwriter.writerow([qtext, qrel])


def make_data(p="../../../datasets/simpledbpediaqa/", embdim=50, save=True):
    train_json = json.load(open(os.path.join(p, "train.json")))
    valid_json = json.load(open(os.path.join(p, "valid.json")))
    test_json = json.load(open(os.path.join(p, "test.json")))

    train_questions = train_json["Questions"]
    valid_questions = valid_json["Questions"]
    test_questions = test_json["Questions"]
    devstart = len(train_questions)
    teststart = len(train_questions) + len(valid_questions)
    allquestions = train_questions + valid_questions + test_questions

    # collect predicates
    predicates = set()
    for question in allquestions:
        question_predicates = question["PredicateList"]
        for pred in question_predicates:
            predicates.add(pred["Predicate"])
    predicates = list(predicates)

    dbp_preds = set()
    dbo_preds = set()
    other_preds = set()
    for predicate in predicates:
        if "dbpedia.org/property" in predicate:
            dbp_preds.add(predicate)
        elif "dbpedia.org/ontology" in predicate:
            dbo_preds.add(predicate)
        else:
            other_preds.add(predicate)

    print("{} unique predicates in {} total questions\n "
          "\t{} property predicates, \n\t{} ontology predicates, \n\t{} other predicates"
          .format(len(predicates), len(allquestions), len(dbp_preds), len(dbo_preds), len(other_preds)))

    for pred in dbp_preds:
        print(pred)

    # get pred strings
    pred_strs = []
    for predicate in predicates:
        if "dbpedia.org/property/" in predicate:
            pred_strs.append(predicate[len("http://dbpedia.org/property/"):])
        elif "dbpedia.org/ontology" in predicate:
            pred_strs.append(predicate[len("http://dbpedia.org/ontology/"):])
        else:
            pred_strs.append("<RARE>")
    psm = q.StringMatrix()
    psm.tokenize = lambda x: [xe.lower() for xe in re.sub('(?!^)([A-Z][a-z]+)', r' \1', x).split()]
    for predstr in pred_strs:
        psm.add(predstr)
    psm.finalize()

    # matrixify questions
    FREQCUTOFF = 3
    sm = q.StringMatrix(freqcutoff=FREQCUTOFF)
    sm.tokenize = lambda x: unidecode.unidecode(x).split()

    for i, question in enumerate(allquestions):
        if i == devstart:
            sm.unseen_mode = True
        sm.add(question["Query"])

    sm.finalize()

    print("{} unique words with freqcutoff {}".format(len(sm.D), FREQCUTOFF))

    # matrixify predicate supervision
    relD = dict(zip(predicates, range(len(predicates))))
    relmat = np.zeros((sm.matrix.shape[0], len(relD)*2), dtype="int64")
    for i, question in enumerate(allquestions):
        for predicate in question["PredicateList"]:
            offset = 0 if predicate["Direction"] == "forward" else len(predicates)
            relmat[i, relD[predicate["Predicate"]] + offset] = 1

    # extend sm.D with all words in glove
    wD = {k: v for k, v in sm.D.items()}
    emb = q.PretrainedWordEmb(embdim)
    for k, v in emb.D.items():
        if k not in wD:
            wD[k] = len(wD)

    # extend qD with all words in psm.D, and map psm.matrix from psm.D to qD
    psmD2qD = {}
    for k, v in psm.D.items():
        if k not in wD:
            wD[k] = len(wD)
        psmD2qD[v] = wD[k]

    relwordmat = np.vectorize(lambda x: psmD2qD[x])(psm.matrix)

    if save:
        pickle.dump([wD, sm.matrix, relwordmat, relmat, relD, (devstart, teststart)],
        open("cache{}.pkl".format(embdim), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    return wD, sm.matrix, relwordmat, relmat, relD, (devstart, teststart)
    """ Data description:
            wD:     word dictionary. contains all of used Glove words + new words found in question + new words found in relation names
            qmat:   question words mapped to ids
            relmat: for every relation name, the word ids of relation
            supmat: supervision matrix. For every example, a row where each relation (first half along axis=1) and its reverse (second half) are 1 if it is an expected relation
            supD:   dictionary mapping from relation names to first half of columns in supmat 
    """


def printrow(row, D):
    revD = {v: k for k, v in D.items()}
    toks = [revD[e] for e in row]
    return " ".join(toks)


class Classifier(torch.nn.Module):
    """ Multiclass classification with embedded and computed labels """
    def __init__(self, inpenc, outdim, outD, numrels, outenc, outdata, **kw):
        super(Classifier, self).__init__(**kw)
        self.inpenc = inpenc
        # for encoding outdata using outenc
        self.outenc = outenc
        self.outdata = outdata
        self.outact = torch.nn.Sigmoid()     # output activation
        self.outvecs_ = q.WordEmb(outdim, worddic=outD)
        self.outvecs_dir_ = q.WordEmb(outdim, worddic={"fwd": 0, "rev": 1})
        self.numrels = numrels

    def forward(self, x):
        """ must return probabilities for every class """
        # do input
        xenc = self.inpenc(x)
        # do labels
        outvecs = torch.cat([self.outvecs_.embedding.weight, self.outvecs_.embedding.weight], 0)
        outvecs_dir = torch.cat([self.outvecs_dir_.embedding.weight[0:1].repeat(self.numrels, 1),
                                 self.outvecs_dir_.embedding.weight[1:2].repeat(self.numrels, 1)],
                                0)
        labelvecs = outvecs + outvecs_dir
        labelvecs_encs = self.outenc(self.outdata)
        labelvecs_encs = torch.cat([labelvecs_encs, labelvecs_encs], 0)
        labelvecs = labelvecs + labelvecs_encs
        # make scores
        scores = torch.einsum("bi,ki->bk", (xenc, labelvecs))
        scores = self.outact(scores)
        return scores


class Encoder(torch.nn.Module):
    def __init__(self, emb, enc, **kw):
        super(Encoder, self).__init__()
        self.emb, self.enc = emb, enc

    def forward(self, x):
        xemb, mask = self.emb(x)
        xenc = self.encqreldir(xemb, mask=mask)
        return xenc


class LSTMEncoder(q.LSTMEncoder):
    def forward(self, x, mask=None):
        out, state = super(LSTMEncoder, self).forward(x, mask=mask, ret_states=True)
        state = state.view(state.size(0), -1)
        return state


class Accuracy(torch.nn.Module):
    def forward(self, y, g):
        yy = y > 0.5
        gg = g > 0.5
        out = (yy == gg).all(1).float()
        out = out.mean()
        return out


def run(lr=0.001,
        glovelr=0.1,
        wordembdim=50,
        encdim=100,
        numlayers=1,
        bidir=False,
        dropout=0.2,
        cuda=False,
        gpu=0,
        epochs=10,
        test=False):
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda", gpu)
    # region data
    tt = q.ticktock("script")
    if not os.path.isfile("cache{}.pkl".format(wordembdim)):
        tt.tick("making data")
        make_data(embdim=wordembdim)
        tt.tock("made data")
    tt.tick("loading data")
    wD, qmat, relmat, supmat, supD, (devstart, teststart) = pickle.load(open("cache{}.pkl".format(wordembdim), "rb"))
    numrels = relmat.shape[0]
    print(numrels)
    tt.tock("loaded data")

    if test:
        print(printrow(qmat[0], wD))
        print(supmat.shape, relmat.shape)
        print(printrow(relmat[115 % relmat.shape[0]], wD))
        revsupD = {v: k for k, v in supD.items()}
        print(np.nonzero(supmat[0]))
    # endregion

    # region model
    wordemb = q.PartiallyPretrainedWordEmb(wordembdim, worddic=wD, gradfracs=(1, glovelr))
    encdims = [encdim] * numlayers
    qenc = LSTMEncoder(wordembdim, *encdims, bidir=bidir, dropout_in=dropout)
    ienc = Encoder(wordemb, qenc)
    outdim = encdims[-1] if not bidir else 2 * encdims[-1]
    oenc = LSTMEncoder(wordembdim, *encdims, bidir=bidir, dropout_in=dropout)
    oenc = Encoder(wordemb, oenc)
    classifier = Classifier(ienc, outdim, supD, numrels, oenc, torch.tensor(relmat))

    if test:
        ys = classifier(torch.tensor(qmat[:10]))
        pass
    # endregion

    loss = torch.nn.BCELoss(reduction="elementwise_mean")
    sup = supmat.astype("float32") * 0.9
    optim = torch.optim.Adam(q.params_of(classifier), lr=lr)

    trainloader = q.dataload(qmat[:devstart], sup[:devstart])
    validloader = q.dataload(qmat[devstart:teststart], sup[devstart:teststart])

    trainer = q.trainer(classifier).on(trainloader).loss(loss).device(device).optimizer(optim)
    validator = q.tester(classifier).on(validloader).loss(loss, Accuracy()).device(device)
    q.train(trainer, validator).run(epochs)


if __name__ == '__main__':
    # q.argprun(run)
    q.argprun(make_data)