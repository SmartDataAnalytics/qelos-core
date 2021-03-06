# -*- coding: utf-8 -*-
import codecs
import json
import os
import torch
import pickle as pkl
import re
from collections import OrderedDict, Counter

import nltk
import numpy as np
import requests

import qelos_core as q
from qelos_core.scripts.query.trees import Node, NodeTrackerDF
from qelos_core.loss import DiscreteLoss
import random
from unidecode import unidecode
from qelos_core.train import BestSaver


class TreeAccuracy(DiscreteLoss):
    def __init__(self, size_average=True, ignore_index=None,
                 treeparser=None, goldgetter=None, **kw):
        """ needs a treeparser that transforms sequences of integer ids to tree objects that support equality
            * treeparser is applied both on x prob's preds (greedy argmax) and the given gold idxs (also when goldgetter is used)
            * goldgetter overrides provided gold with other gold (gets pred probs and provided gold)
        """
        super(TreeAccuracy, self).__init__(size_average=size_average, ignore_index=ignore_index, **kw)
        self.treeparser = treeparser
        self.goldgetter = goldgetter

    def forward(self, x, gold, mask=None):
        if mask is not None and mask.data[0, 1] > 1:  # batchable sparse
            assert(False)   # no support for this anymore
            # mask = q.batchablesparse2densemask(mask)
        if mask is not None:
            x = x + torch.log(mask.float())

        if self.goldgetter is not None:
            gold = self.goldgetter(x, gold)

        ignoremask = self._get_ignore_mask(gold)
        maxes, best = torch.max(x, 2)
        same = torch.zeros(best.size(0), dtype=torch.float32, device=x.device)
        for i in range(best.size(0)):
            try:
                best_tree = self.treeparser(best[i].cpu().data.numpy())
            except Exception as e:
                best_tree = None
            gold_tree = self.treeparser(gold[i].cpu().data.numpy())
            if isinstance(best_tree, tuple):
                print(best_tree)
            same[i] = 1 if gold_tree.equals(best_tree) else 0
        acc = torch.sum(same)
        total = float(same.size(0))
        if self.size_average:
            acc = acc / total
        # if ignoremask is not None:
        #     same.data = same.data | ~ ignoremask.data
        return acc


# TODO: !!!!!!!!!!!!! DON'T FORGET THAT VALID BT'S OVERRIDE NORMAL BT'S --> HAVE TO SET TO NONE TO CANCEL TRAIN TIME BT
#       --> TODO: CHECK IN OLD SCRIPT AND NEW SCRIPT THAT NOTHING WRONG HAPPENS !!!!!!!!!

# REMARK:   changed set_(valid)_batch_transform() to accept None's as they are
#           before, might have "sticky" bt's, but shouldn't have affected old wikisql TODO: CHECK !!!

# TODO: make sure test and dev splits are correct
# TODO: MAKE SURE vanilla embeddings are changed after training

# TODO: rare words in col names
# DONE: UNIQUE RARES  --> make_inp_emb and OutVecComputer use rare-X embeddings based on rare_gwids_not_in_glove
#                         (implemented by a hack replacement with rare_vec)


_opt_test = True
DATA_PATH = "../../../datasets/wikisql_clean/"


# region DATA
# region GENERATING .LINES
def read_jsonl(p):
    """ takes a path and returns objects from json lines """
    lines = []
    with open(p) as f:
        for line in f:
            example = json.loads(line)
            lines.append(example)
    return lines


def jsonls_to_lines(p=DATA_PATH):
    """ loads all jsons, converts them to .lines, saves and returns """
    # region load all jsons
    tt = q.ticktock("data preparer")
    tt.tick("loading jsons")

    # load examples
    traindata = read_jsonl(p+"train.jsonl")
    print("{} training questions".format(len(traindata)))
    traindb = read_jsonl(p+"train.tables.jsonl")
    print("{} training tables".format(len(traindb)))
    devdata = read_jsonl(p + "dev.jsonl")
    print("{} dev questions".format(len(devdata)))
    devdb = read_jsonl(p+"dev.tables.jsonl")
    print("{} dev tables".format(len(devdb)))
    testdata = read_jsonl(p + "test.jsonl")
    print("{} test questions".format(len(testdata)))
    testdb = read_jsonl(p + "test.tables.jsonl")
    print("{} test tables".format(len(testdb)))

    # join all tables in one
    alltables = {}
    for table in traindb + devdb + testdb:
        alltables[table["id"]] = table

    print("total number of tables: {} ".format(len(alltables)))
    tt.tock("jsons loaded")
    # endregion

    # coltypes, tD, (types_devstart, types_teststart) = get_column_types(traindata, devdata, testdata, alltables)     # "real" = 1, "text" = 2
    # coltypes = list(coltypes)
    # _coltypes = []
    coltypes = []
    types_maxlen = 0
    tD = {"text": 2, "real": 1}

    # region generating examples
    tt.tick("generating examples")
    tt.msg("train examples")
    trainexamples = []
    for line in traindata:
        try:
            trainexamples.append(jsonl_to_line(line, alltables))
            column_types = alltables[line["table_id"]]["types"]
            types_maxlen = max(types_maxlen, len(column_types))
            coltypes.append(column_types)
        except Exception as e:
            print("FAILED: {}".format(line))

    # IMPORTANT: try not to omit any examples in dev and test
    tt.msg("dev examples")
    devexamples = []
    for line in devdata:
        try:
            devexamples.append(jsonl_to_line(line, alltables))
            column_types = alltables[line["table_id"]]["types"]
            types_maxlen = max(types_maxlen, len(column_types))
            coltypes.append(column_types)
        except Exception as e:
            print("FAILED: {}".format(line))

    tt.msg("test examples")
    testexamples = []
    for line in testdata:
        try:
            testexamples.append(jsonl_to_line(line, alltables))
            column_types = alltables[line["table_id"]]["types"]
            types_maxlen = max(types_maxlen, len(column_types))
            coltypes.append(column_types)
        except Exception as e:
            print("FAILED: {}".format(line))

    tt.tock("examples generated")
    # endregion

    typesmat = np.zeros((len(coltypes), types_maxlen), dtype="int32")
    for i, types in enumerate(coltypes):
        for j, t in enumerate(types):
            typesmat[i, j] = tD[t]
    np.save(p + "coltypes.mat", typesmat)

    # region save lines
    tt.msg("saving lines")
    print("\n".join(trainexamples[:10]))

    with codecs.open(p + "train.lines", "w", encoding="utf-8") as f:
        for example in trainexamples:
            f.write("{}\n".format(example))
    with codecs.open(p + "dev.lines", "w", encoding="utf-8") as f:
        for example in devexamples:
            f.write("{}\n".format(example))
    with codecs.open(p + "test.lines", "w", encoding="utf-8") as f:
        for example in testexamples:
            f.write("{}\n".format(example))
    # endregion

    return trainexamples, devexamples, testexamples


def jsonl_to_line(line, alltables):
    """ takes object from .jsonl, creates line"""
    column_names = alltables[line["table_id"]]["header"]
    question = " ".join(nltk.word_tokenize(line["question"])).lower()

    # region replacements on top of tokenization:
    # question = question.replace(u"\u2009", u" ")    # unicode long space to normal space
    # question = question.replace(u"\u2013", u"-")    # unicode long dash to normal dash
    question = question.replace("`", "'")         # unicode backward apostrophe to normal
    question = question.replace("\u00b4", "'")    # unicode forward apostrophe to normal
    question = question.replace("''", '"')    # double apostrophe to quote
    # add spaces around some special characters because nltk tokenizer doesn't:
    question = question.replace('\u20ac', ' \u20ac ')     # euro sign
    question = question.replace('\uffe5', ' \uffe5 ')     # yen sign
    question = question.replace('\u00a3', ' \u00a3 ')     # pound sign
    question = question.replace("'", " ' ")
    question = re.sub('/$', ' /', question)
    # split up meters and kilometers because nltk tokenizer doesn't
    question = re.sub('(\d+[,\.]\d+)(k?m)', '\g<1> \g<2>', question)

    question = re.sub('\s+', ' ', question)
    # endregion

    # MANUAL FIXES FOR TYPOS OR FAILED TOKENIZATION (done in original jsonl's) - train fixes not included:
    # dev, line 6277, "25,000." to "25,000 ,"
    # dev, line 7784, "No- Gold Coast" to "No Gold Coast"
    # test, line 6910, "Difference of- 17" to "Difference of - 17" (added space)
    # test, line 2338, "baccalaureate colleges" to "baccalaureate college" (condition value contains latter)
    # test, line 5440, "44+" to "44 +"
    # test, line 7159, "a frequency of 1600MHz and voltage under 1.35V" to "a frequency of 1600 MHz and voltage under 1.35 V" (added two spaces) and changed condition "1600mhz" to "1600 mhz"
    # test, line 8042, "under 20.6bil" to "under 20.6 bil" (added space)
    # test, line 8863, replaced long spaces "\u2009" with normal space
    # test, line 8866, replaced long spaces "\u2009" with normal space
    # test, line 8867, replaced long spaces "\u2009" with normal space
    # test, line 13290, "35.666sqmi" to "35.666 sqmi" (added space)
    # BAD TEST CHANGES (left in to ensure consistency of line numbers)
    # test, line 6077, changed first condition from "\u2013" to "no"

    # region construct query
    # select clause
    sql_select = "AGG{} COL{}".format(line["sql"]["agg"], line["sql"]["sel"])
    # where clause
    sql_wheres = []
    for cond in line["sql"]["conds"]:
        if isinstance(cond[2], float):
            condval = str(cond[2].__repr__())       # printing float in original precision
        else:
            condval = str(cond[2]).lower()

        # replacements in condval, equivalent to question replacements
        # condval = condval.replace(u"\u2009", u" ")    # unicode long space to normal space
        # condval = condval.replace(u"\u2013", u"-")    # unicode long dash to normal dash
        condval = condval.replace("`", "'")
        condval = condval.replace("\u00b4", "'")    # unicode forward apostrophe to normal
        condval = condval.replace("''", '"')
        _condval = condval.replace(" ", "")

        # region rephrase condval in terms of span of question
        condval = None
        questionsplit = question.split()

        for i, qword in enumerate(questionsplit):
            for qwordd in _condval:
                found = False
                for j in range(i+1, len(questionsplit) + 1):
                    if "".join(questionsplit[i:j]) in _condval:
                        if "".join(questionsplit[i:j]) == _condval:
                            condval = " ".join(questionsplit[i:j])
                            found = True
                            break
                    else:
                        break
                if found:
                    break
        assert(condval in question)
        # endregion

        condl = "<COND> COL{} OP{} <VAL> {} <ENDVAL>".format(cond[0], cond[1], condval)
        sql_wheres.append(condl)

    # create full query:
    if len(sql_wheres) > 0:
        sql = "<QUERY> <SELECT> {} <WHERE> {}".format(sql_select, " ".join(sql_wheres))
    else:
        sql = "<QUERY> <SELECT> {}".format(sql_select)
    ret = "{}\t{}\t{}".format(question, sql, "\t".join(column_names))
    # endregion
    return ret


def load_coltypes(p=DATA_PATH):
    return np.load(p+"coltypes.mat.npy")
# endregion


# region GENERATING .MATS
def load_lines(p):
    retlines = []
    with codecs.open(p, encoding="utf-8") as f:
        for line in f:
            linesplits = line.strip().split("\t")
            assert(len(linesplits) > 2)
            retlines.append((linesplits[0], linesplits[1], linesplits[2:]))
    return retlines


def create_mats(p=DATA_PATH):
    # loading lines
    tt = q.ticktock("data loader")
    tt.tick("loading lines")
    trainlines = load_lines(p+"train.lines")
    print("{} train lines".format(len(trainlines)))
    devlines = load_lines(p+"dev.lines")
    print("{} dev lines".format(len(devlines)))
    testlines = load_lines(p+"test.lines")
    print("{} test lines".format(len(testlines)))
    tt.tock("lines loaded")

    # preparing matrices
    tt.tick("preparing matrices")
    i = 0
    devstart, teststart = 0, 0

    # region gather original dictionary
    ism = q.StringMatrix()
    ism.tokenize = lambda x: x.split()
    numberunique = 0
    for question, query, columns in trainlines:
        numberunique = max(numberunique, len(set(question.split())))
        ism.add(question)
        i += 1
    devstart = i
    for question, query, columns in devlines:
        numberunique = max(numberunique, len(set(question.split())))
        ism.add(question)
        i += 1
    teststart = i
    for question, query, columns in testlines:
        numberunique = max(numberunique, len(set(question.split())))
        ism.add(question)
        i += 1
    ism.finalize()
    print("max number unique words in a question: {}".format(numberunique))
    # endregion

    gwids = np.ones((ism.matrix.shape[0], numberunique + 3),
                     dtype="int64")  # per-example dictionary, mapping UWID position to GWID
                                     # , mapping unused UWIDs to <RARE> GWID

    # ism.D contains dictionary over all question words
    gwids = gwids * ism.D["<RARE>"]  # set default to gwid <RARE>
    gwids[:, 0] = ism.D["<MASK>"]  # set UWID0 to <MASK> for all examples

    # pedics matrix is used as follows:
    # suppose question has "UWID1 UWID2", then map to ints ([1, 2]),
    # select [gwids[example_id, 1], gwids[example_id, 2]] to get the actual words

    uwids = np.zeros_like(ism.matrix)                         # questions in terms of uwids

    rD = {v: k for k, v in ism.D.items()}                     # gwid reverse dictionary

    gwid2uwid_dics = []           # list of dictionaries mapping gwids to uwids for every example

    for i in range(len(ism.matrix)):    # for every example
        row = ism.matrix[i]             # get row
        gwid2uwid = {"<MASK>": 0}      # initialize gwid2uwid dictionary (for this example)
        for j in range(len(row)):       # for position in row
            k = row[j]                  # get gwid for word at that position
            if rD[k] not in gwid2uwid:                  # word for that gwid is not in gwid2uwid
                gwid2uwid[rD[k]] = len(gwid2uwid)       # add uwid for the word for that gwid to gwid2wid
                gwids[i, gwid2uwid[rD[k]]] = k          # add mapping from new uwid to gwid
            uwids[i, j] = gwid2uwid[rD[k]]              # put uwid in uwid mat
        gwid2uwid_dics.append(gwid2uwid)                # add gwid2uwid dic to list

    # create dictionary from uwid words to ids
    uwidD = dict(
        zip(["<MASK>"] + ["UWID{}".format(i + 1) for i in range(gwids.shape[1] - 1)],
            range(gwids.shape[1])))

    # region target sequences matrix
    osm = q.StringMatrix(indicate_start=True, indicate_end=True)
    osm.tokenize = lambda x: x.split()
    i = 0
    for _, query, columns in trainlines + devlines + testlines:
        query_tokens = query.split()
        gwid2uwid = gwid2uwid_dics[i]
        query_tokens = ["UWID{}".format(gwid2uwid[e])       # map target words to UWIDs where possible
                        if e in gwid2uwid else e
                        for e in query_tokens]
        _q = " ".join(query_tokens)
        osm.add(_q)
        i += 1
    osm.finalize()
    # endregion

    # region column names
    example2columnnames = np.zeros((len(osm), 44), dtype="int64")
                # each row contains sequence of column name ids available for that example
    uniquecolnames = OrderedDict({"nonecolumnnonecolumnnonecolumn": 0})     # unique column names
    i = 0
    for _, query, columns in trainlines + devlines + testlines:
        j = 0
        for column in columns:
            if column not in uniquecolnames:
                uniquecolnames[column] = len(uniquecolnames)
            example2columnnames[i, j] = uniquecolnames[column]
            j += 1
        i += 1

    # create matrix with for every unique column name (row), the sequence of words describing it
    csm = q.StringMatrix(indicate_start=False, indicate_end=False)
    for i, columnname in enumerate(uniquecolnames.keys()):
        if columnname == '№':
            columnname = 'number'
        csm.add(columnname)
    csm.finalize()
    # idx 3986 is zero because it's u'№' and unidecode makes it empty string, has 30+ occurrences
    assert(len(np.argwhere(csm.matrix[:, 0] == 0)) == 0)
    # endregion

    # region save
    with open(p + "matcache.mats", "w") as f:
        np.savez(f, ism=uwids, osm=osm.matrix, csm=csm.matrix, pedics=gwids, e2cn=example2columnnames)
    with open(p + "matcache.dics", "w") as f:
        dics = {"ism": uwidD, "osm": osm.D, "csm": csm.D, "pedics": ism.D, "sizes": (devstart, teststart)}
        pkl.dump(dics, f, protocol=pkl.HIGHEST_PROTOCOL)

    # print("question dic size: {}".format(len(ism.D)))
    # print("question matrix size: {}".format(ism.matrix.shape))
    # print("query dic size: {}".format(len(osm.D)))
    # print("query matrix size: {}".format(osm.matrix.shape))
    tt.tock("matrices prepared")
    # endregion


def load_matrices(p=DATA_PATH):
    """ loads matrices generated before.
        Returns:    * ism: input questions - in uwids
                    * osm: target sequences - use uwids for words
                    * csm: column names for unique column names
                    * gwids: for every uwid in ism/osm, mapping to gwids by position
                    * splits: indexes where train ends and dev ends
                    * e2cn: example ids to column names mapping (matrix)

        """
    tt = q.ticktock("matrix loader")
    tt.tick("loading matrices")
    with open(p+"matcache.mats") as f:
        mats = np.load(f)
        ismmat, osmmat, csmmat, gwidsmat, e2cn \
            = mats["ism"], mats["osm"], mats["csm"], mats["pedics"], mats["e2cn"]
    tt.tock("matrices loaded")
    print(ismmat.shape)
    tt.tick("loading dics")
    with open(p+"matcache.dics") as f:
        dics = pkl.load(f)
        ismD, osmD, csmD, pedicsD, splits = dics["ism"], dics["osm"], dics["csm"], dics["pedics"], dics["sizes"]
    tt.tock("dics loaded")
    # ensure that osmD contains all items from ismD
    addeduwids = set()
    for k, v in ismD.items():
        if k not in osmD:
            osmD[k] = max(osmD.values()) + 1
            addeduwids.add(k)

    # add UWID0 to osmD
    osmD["UWID0"] = max(osmD.values()) + 1
    addeduwids.add("UWID0")
    print("added {} uwids: {}".format(len(addeduwids), " ".join(list(addeduwids))))

    # add all remaining COL-ids to osmD
    allcolids = set([int(re.match("COL(\d+)", x).group(1)) for x in osmD.keys() if re.match("COL\d+", x)])
    addedcolids = set()
    for i in range(e2cn.shape[1]):
        if i not in allcolids:
            osmD["COL{}".format(i)] = max(osmD.values()) + 1
            addedcolids.add("COL{}".format(i))

    print("added {} colids: {}".format(len(addedcolids), " ".join(list(addedcolids))))

    print(len(ismD))
    ism = q.StringMatrix()
    ism.set_dictionary(ismD)
    ism._matrix = ismmat
    osm = q.StringMatrix()
    osm.set_dictionary(osmD)
    osm._matrix = osmmat
    csm = q.StringMatrix()
    csm.set_dictionary(csmD)
    csm._matrix = csmmat
    gwids = q.StringMatrix()
    gwids.set_dictionary(pedicsD)
    gwids._matrix = gwidsmat
    # q.embed()
    return ism, osm, csm, gwids, splits, e2cn


def reconstruct_question(uwids, gwids, rgd):
    words = gwids[uwids]
    question = " ".join([rgd[wordid] for wordid in words])
    question = question.replace("<MASK>", " ")
    question = re.sub("\s+", " ", question)
    question = question.strip()
    return question


def reconstruct_query(osmrow, gwidrow, rod, rgd):
    query = " ".join([rod[wordid] for wordid in osmrow])
    query = query.replace("<MASK>", " ")
    query = re.sub("\s+", " ", query)
    query = query.strip()
    query = re.sub("UWID\d+", lambda x: rgd[gwidrow[int(x.group(0)[4:])]], query)
    return query


# region test
def tst_matrices(p=DATA_PATH, writeout=False):
    ism, osm, csm, gwids, splits, e2cn = load_matrices()
    devlines = load_lines(p+"dev.lines")
    print("{} dev lines".format(len(devlines)))
    testlines = load_lines(p+"test.lines")
    print("{} test lines".format(len(testlines)))
    devstart, teststart = splits

    dev_ism, dev_gwids, dev_osm, dev_e2cn = ism.matrix[devstart:teststart], gwids.matrix[devstart:teststart], \
                                            osm.matrix[devstart:teststart], e2cn[devstart:teststart]
    test_ism, test_gwids, test_osm, test_e2cn = ism.matrix[teststart:], gwids.matrix[teststart:], \
                                                osm.matrix[teststart:], e2cn[teststart:]
    rgd = {v: k for k, v in gwids.D.items()}
    rod = {v: k for k, v in osm.D.items()}

    # test question reconstruction
    for i in range(len(devlines)):
        orig_question = devlines[i][0].strip()
        reco_question = reconstruct_question(dev_ism[i], dev_gwids[i], rgd)
        assert(orig_question == reco_question)
    print("dev questions reconstruction matches")
    for i in range(len(testlines)):
        orig_question = testlines[i][0].strip()
        reco_question = reconstruct_question(test_ism[i], test_gwids[i], rgd)
        assert(orig_question == reco_question)
    print("test questions reconstruction matches")

    # test query reconstruction
    dev_reco_queries = []
    for i in range(len(devlines)):
        orig_query = devlines[i][1].strip()
        reco_query = reconstruct_query(dev_osm[i], dev_gwids[i], rod, rgd).replace("<START>", "").replace("<END>", "").strip()
        try:
            assert (orig_query == reco_query)
        except Exception as e:
            print("FAILED: {} \n - {}".format(orig_query, reco_query))
        dev_reco_queries.append(reco_query)
    print("dev queries reconstruction matches")
    test_reco_queries = []
    for i in range(len(testlines)):
        orig_query = testlines[i][1].strip()
        reco_query = reconstruct_query(test_osm[i], test_gwids[i], rod, rgd).replace("<START>", "").replace("<END>", "").strip()
        assert (orig_query == reco_query)
        test_reco_queries.append(reco_query)
    print("test queries reconstruction matches")

    if writeout:
        with codecs.open(DATA_PATH+"dev.gold.outlines", "w", encoding="utf-8") as f:
            for query in dev_reco_queries:
                f.write("{}\n".format(query))
        with codecs.open(DATA_PATH+"test.gold.outlines", "w", encoding="utf-8") as f:
            for query in test_reco_queries:
                f.write("{}\n".format(query))

# endregion
# endregion

# endregion

# region SQL TREES
# region Node and Order
class SqlNode(Node):
    mode = "limited"
    name2ctrl = {
        "<SELECT>": "A",
        "<WHERE>":  "A",
        "<COND>":   "A",
        "COL\d+":   "NC",
        "AGG\d+":   "NC",
        "OP\d+":    "NC",
        "<VAL>":    "A",
        "<ENDVAL>": "NC",
        "UWID\d+":  "NC",
    }

    def __init__(self, name, order=None, children=tuple(), **kw):
        super(SqlNode, self).__init__(name, order=order, children=children, **kw)

    def __eq__(self, other):
        return super(SqlNode, self).__eq__(other)

    @classmethod
    def parse_sql(cls, inp, _rec_arg=None, _toprec=True, _ret_remainder=False):
        """ ONLY FOR ORIGINAL LIN
            * Automatically assigns order to children of <VAL> !!! """
        if len(inp) == 0:
            return []
        tokens = inp
        parent = _rec_arg
        if _toprec:
            tokens = tokens.replace("  ", " ").strip().split()
        head = tokens[0]
        tail = tokens[1:]

        siblings = []
        jumpers = {"<SELECT>": {"<QUERY>"},
                   "<WHERE>": {"<QUERY>"},
                   "<COND>": {"<WHERE>"},
                   "<VAL>": {"<COND>"},}
        while True:
            head, islast, isleaf = head, None, None

            # TODO: might want to disable this
            headsplits = head.split(SqlNode.suffix_sep)
            if len(headsplits) in (2, 3):
                if headsplits[1] in (SqlNode.leaf_suffix, SqlNode.last_suffix) \
                    and (len(headsplits) == 1 or (headsplits[2] in (SqlNode.leaf_suffix, SqlNode.last_suffix))):
                        head, islast, isleaf = headsplits[0], SqlNode.last_suffix in headsplits, SqlNode.leaf_suffix in headsplits

            if head == "<QUERY>":
                assert (isleaf is None or isleaf is False)
                assert (islast is None or islast is True)
                children, tail = cls.parse_sql(tail, _rec_arg=head, _toprec=False)
                ret = SqlNode(head, children=children)
                break
            elif head == "<END>":
                ret = siblings, []
                break
            elif head in jumpers:
                assert (isleaf is None or isleaf is False)
                if _rec_arg in jumpers[head]:
                    children, tail = cls.parse_sql(tail, _rec_arg=head, _toprec=False)
                    if head == "<VAL>":
                        for i, child in enumerate(children):
                            child.order = i
                    node = SqlNode(head, children=children)
                    siblings.append(node)
                    if len(tail) > 0:
                        head, tail = tail[0], tail[1:]
                    else:
                        ret = siblings, tail
                        break
                else:
                    ret = siblings, [head] + tail
                    break
            else:
                assert (isleaf is None or isleaf is True)
                node = SqlNode(head)
                siblings.append(node)
                if head == "<ENDVAL>" or len(tail) == 0:
                    ret = siblings, tail
                    break
                else:
                    head, tail = tail[0], tail[1:]
        if isinstance(ret, tuple):
            if _toprec:
                raise q.SumTingWongException("didn't parse SQL in .parse_sql()")
            else:
                return ret
        else:
            if cls.mode == "limited":
                order_adder_wikisql_limited(ret)
            else:
                order_adder_wikisql(ret)
            return ret

    @classmethod
    def parse(cls, inp, _rec_arg=None, _toprec=True, _ret_remainder=False):
        tokens = inp
        if _toprec:
            tokens = tokens.replace("  ", " ").strip().split()
            for i in range(len(tokens)):
                splits = tokens[i].split("*")
                if len(splits) == 1:
                    token, suffix = splits[0], ""
                else:
                    token, suffix = splits[0], "*" + splits[1]
                if token not in "<QUERY> <SELECT> <WHERE> <COND> <VAL>".split():
                    suffix = "*NC" + suffix
                tokens[i] = token + suffix
            ret = super(SqlNode, cls).parse(" ".join(tokens), _rec_arg=None, _toprec=True)
            if cls.mode == "limited":
                order_adder_wikisql_limited(ret)
            else:
                order_adder_wikisql(ret)
            return ret
        else:
            return super(SqlNode, cls).parse(tokens, _rec_arg=_rec_arg, _toprec=_toprec)

    @classmethod
    def parse_df(cls, inp, _toprec=True):
        if _toprec:
            parse = super(SqlNode, cls).parse_df(inp, _toprec=_toprec)
            if cls.mode == "limited":
                order_adder_wikisql_limited(parse)
            else:
                order_adder_wikisql(parse)
            return parse
        else:
            return super(SqlNode, cls).parse_df(inp, _toprec)

    def pp_sql(self, arbitrary=False):
        children = list(self.children)

        if arbitrary is True:
            # randomly shuffle children while keeping children with order in positions they were in
            fillthis = [child if child._order is not None else None for child in children]
            if None in fillthis:
                pass
            children_without_order = [child for child in children if child._order is None]
            random.shuffle(children_without_order)
            for i in range(len(fillthis)):
                if fillthis[i] is None:
                    fillthis[i] = children_without_order[0]
                    children_without_order = children_without_order[1:]
            children = fillthis

        children = [child.pp_sql(arbitrary=arbitrary)
                    for child, _is_last_child
                    in zip(children, [False] * (len(children) - 1) + [True])]
        ret = self.symbol(with_label=True, with_annotation=False, with_order=False)
        ret += "" if len(children) == 0 else " " + " ".join(children)
        return ret


def get_children_by_name(node, cre):
    for child in node.children:
        if re.match(cre, child.name):
            yield child


def querylin2json(qlin, origquestion):
    try:
        parsedtree = SqlNode.parse_sql(qlin)
        assert(parsedtree.name == "<QUERY>")            # root must by <query>
        # get select and where subtrees
        selectnode = list(get_children_by_name(parsedtree, "<SELECT>"))
        assert(len(selectnode) == 1)
        selectnode = selectnode[0]
        wherenode = list(get_children_by_name(parsedtree, "<WHERE>"))
        assert(len(wherenode) <= 1)
        if len(wherenode) == 0:
            wherenode = None
        else:
            wherenode = wherenode[0]
        assert(selectnode.name == "<SELECT>")
        assert(wherenode is None or wherenode.name == "<WHERE>")
        # get select arguments
        assert(len(selectnode.children) == 2)
        select_col = list(get_children_by_name(selectnode, "COL\d{1,2}"))
        assert(len(select_col) == 1)
        select_col = int(select_col[0].name[3:])
        select_agg = list(get_children_by_name(selectnode, "AGG\d"))
        assert(len(select_agg) == 1)
        select_agg = int(select_agg[0].name[3:])
        # get where conditions
        conds = []
        if wherenode is not None:
            for child in wherenode.children:
                assert(child.name == "<COND>")
                assert(len(child.children) == 3)
                cond_col = list(get_children_by_name(child, "COL\d{1,2}"))
                assert(len(cond_col) == 1)
                cond_col = int(cond_col[0].name[3:])
                cond_op = list(get_children_by_name(child, "OP\d"))
                assert(len(cond_op) == 1)
                cond_op = int(cond_op[0].name[2:])
                val_node = list(get_children_by_name(child, "<VAL>"))
                assert(len(val_node) == 1)
                val_node = val_node[0]
                val_nodes = val_node.children
                if val_nodes[-1].name == "<ENDVAL>":        # all should end with endval but if not, accept
                    val_nodes = val_nodes[:-1]
                valstring = " ".join([x.name for x in val_nodes])
                valsearch = re.escape(valstring.lower()).replace("\\ ", "\s?")
                found = re.findall(valsearch, origquestion.lower())
                if len(found) > 0:
                    found = found[0]
                    conds.append([cond_col, cond_op, found])
        return {"sel": select_col, "agg": select_agg, "conds": conds}
    except Exception as e:
        return {"agg": 10, "sel": 1345, "conds": [[5, 0, "https://www.youtube.com/watch?v=oHg5SJYRHA0"]]}


def same_sql_json(x, y):    # should not matter whether x or y is pred or gold, better to have x as gold
    same = True
    same &= x["sel"] == y["sel"]
    same &= x["agg"] == y["agg"]
    same &= len(x["conds"]) == len(y["conds"])
    xconds = x["conds"] + []
    yconds = y["conds"] + []
    for xcond in xconds:
        found = False
        for j in range(len(yconds)):
            xcondval = xcond[2]
            if isinstance(xcondval, float):
                xcondval = xcondval.__repr__()
            ycondval = yconds[j][2]
            if isinstance(ycondval, float):
                ycondval = ycondval.__repr__()
            xcondval, ycondval = str(xcondval), str(ycondval)
            if xcond[0] == yconds[j][0] \
                    and xcond[1] == yconds[j][1] \
                    and xcondval.lower() == ycondval.lower():
                found = True
                del yconds[j]
                break
        same &= found
    if same:
        assert(len(yconds) == 0)
    return same


def same_lin_json(x, y):
    same = True
    same &= x["sel"] == y["sel"]
    same &= x["agg"] == y["agg"]
    same &= len(x["conds"]) == len(y["conds"])
    xconds = x["conds"] + []
    yconds = y["conds"] + []
    if same:
        for xcond, ycond in zip(xconds, yconds):
            xcondval = xcond[2]
            if isinstance(xcondval, float):
                xcondval = xcondval.__repr__()
            ycondval = ycond[2]
            if isinstance(ycondval, float):
                ycondval = ycondval.__repr__()
            xcondval, ycondval = str(xcondval), str(ycondval)
            if xcond[0] == ycond[0] \
                    and xcond[1] == ycond[1] \
                    and xcondval.lower() == ycondval.lower():
                same &= True
            else:
                same &= False
    return same


def load_jsonls(p, questionsonly=False, sqlsonly=False):
    ret = []
    with open(p) as f:
        for line in f:
            jsonl = json.loads(line)
            if questionsonly:
                question = jsonl["question"]
                ret.append(question)
            elif sqlsonly:
                sql = jsonl["sql"]
                ret.append(sql)
            else:
                ret.append(jsonl)
    return ret


def tst_querylin2json():
    qlin = "<QUERY> <SELECT> AGG0 COL3 <WHERE> <COND> COL5 OP0 <VAL> butler cc ( ks ) <ENDVAL>"
    jsonl = """{"phase": 1, "table_id": "1-10015132-11", "question": "What position does the player who played for butler cc (ks) play?",
                "sql": {"sel": 3, "conds": [[5, 0, "Butler CC (KS)"]], "agg": 0}}"""
    jsonl = json.loads(jsonl)
    origquestion = jsonl["question"]
    orig_sql = jsonl["sql"]
    recon_sql = querylin2json(qlin, origquestion)
    assert(same_sql_json(recon_sql, orig_sql))

    # test dev lines
    p = DATA_PATH
    devlines = load_lines(p + "dev.lines")
    failures = 0
    with open(p + "dev.jsonl") as f:
        i = 0
        for l in f:
            jsonl = json.loads(l)
            origquestion, orig_sql = jsonl["question"], jsonl["sql"]
            recon_sql = querylin2json(devlines[i][1], origquestion)
            try:
                assert (same_sql_json(recon_sql, orig_sql))
            except Exception as e:
                failures += 1
                print("FAILED: {}: {}\n-{}".format(i, orig_sql, recon_sql))
            i += 1
    if failures == 0:
        print("dev querylin2json passed")
    else:
        print("dev querylin2json: FAILED")

    # test test lines
    p = DATA_PATH
    devlines = load_lines(p+"test.lines")
    failures = 0
    with open(p+"test.jsonl") as f:
        i = 0
        for l in f:
            jsonl = json.loads(l)
            origquestion, orig_sql = jsonl["question"], jsonl["sql"]
            recon_sql = querylin2json(devlines[i][1], origquestion)
            try:
                assert(same_sql_json(recon_sql, orig_sql))
            except Exception as e:
                failures += 1
                print("FAILED: {}: {}\n-{}".format(i, orig_sql, recon_sql))
            i += 1
    if failures == 0:
        print("test querylin2json passed")
    else:
        print("test querylin2json: FAILED - expected 1 failure")
    # !!! example 15485 in test fails because wrong gold constructed in lines (wrong occurence of 2-0 is taken)


def order_adder_wikisql(parse):
    # add order to children of VAL
    def order_adder_rec(y):
        for i, ychild in enumerate(y.children):
            if y.name == "<VAL>":
                ychild.order = i
            order_adder_rec(ychild)

    order_adder_rec(parse)
    return parse


def order_adder_wikisql_limited(parse):
    # add order everywhere except children of WHERE
    def order_adder_rec(y):
        for i, ychild in enumerate(y.children):
            if y.name != "<WHERE>":
                ychild.order = i
            order_adder_rec(ychild)

    order_adder_rec(parse)
    return parse


# region test
def tst_sqlnode_and_sqls(x=0):
    orig_question = "which city of license has a frequency mhz smaller than 100.9 , and a erp w larger than 100 ?"
    orig_line = "<QUERY> <SELECT> AGG0 COL2 " \
                "<WHERE> <COND> COL1 OP2 <VAL> 100.9 <ENDVAL> <COND> COL3 OP1 <VAL> 100 <ENDVAL>"
    orig_tree = SqlNode.parse_sql(orig_line)
    orig_sql = querylin2json(orig_line, orig_question)

    print(orig_tree.pptree())

    swapped_conds_line = "<QUERY> <SELECT> AGG0 COL2 " \
                "<WHERE> <COND> COL3 OP1 <VAL> 100 <ENDVAL> <COND> COL1 OP2 <VAL> 100.9 <ENDVAL>"

    swapped_conds_tree = SqlNode.parse_sql(swapped_conds_line)
    swapped_conds_sql = querylin2json(swapped_conds_line, orig_question)

    print("swapped conds testing:")
    assert(orig_tree == swapped_conds_tree)
    print("trees same")
    assert(same_sql_json(orig_sql, swapped_conds_sql))
    print("sqls same")

    swapped_select_args_line = "<QUERY> <SELECT> COL2 AGG0 " \
                "<WHERE> <COND> COL1 OP2 <VAL> 100.9 <ENDVAL> <COND> COL3 OP1 <VAL> 100 <ENDVAL>"
    swapped_select_args_tree = SqlNode.parse_sql(swapped_select_args_line)
    swapped_select_args_sql = querylin2json(swapped_select_args_line, orig_question)
    print("swapped select args testing:")
    assert(orig_tree != swapped_select_args_tree)
    print("trees NOT same")
    assert(same_sql_json(orig_sql, swapped_select_args_sql))
    print("sqls same")

    wrong_line = "<QUERY> <SELECT> AGG0 COL2 " \
                "<WHERE> <COND> COL1 OP2 <VAL> 100.92 <ENDVAL> <COND> COL3 OP1 <VAL> 100 <ENDVAL>"
    wrong_tree = SqlNode.parse_sql(wrong_line)
    wrong_sql = querylin2json(wrong_line, orig_question)
    print("wrong query testing:")
    assert(orig_tree != wrong_tree)
    print("trees NOT same")
    assert(not same_sql_json(orig_sql, wrong_sql))
    print("sqls NOT same")

    bad_line = "<QUERY> <SELECT> AGG0 COL2 " \
                "<WHERE> <COND> COL1 OP2 <VAL> 100.9 <ENDVAL> <COND> COL3 OP1 <VAL> 100 "
    bad_tree = SqlNode.parse_sql(bad_line)
    print("bad correct line: tree parsed")
    bad_sql = querylin2json(bad_line, orig_question)
    assert(same_sql_json(orig_sql, bad_sql))
    print("sqls same")

    # test with broken lines
    linesplits = orig_line.split()
    for i in range(len(linesplits)-1):      # every except last one
        for j in range(i+1, len(linesplits)):
            broken_line = linesplits[:i] + linesplits[j:]
            broken_line = " ".join(broken_line)
            try:
                broken_tree = SqlNode.parse_sql(broken_line)
                broken_sql = querylin2json(broken_line, orig_question)
                assert(broken_tree != orig_tree)
                if " ".join(linesplits[i:j]) not in ("<ENDVAL>"):
                    assert(not same_sql_json(broken_sql, orig_sql))
            except q.SumTingWongException as e:
                # didn't parse
                pass
    print("all brokens passed")



    # TODO: test parsing from different linearizations
    # TODO: test __eq__
    # TODO: test order while parsing and __eq__
    pass
# endregion

# endregion

# region Trackers
class SqlGroupTrackerDF(object):
    def __init__(self, trackables, coreD):
        super(SqlGroupTrackerDF, self).__init__()
        self.trackables = trackables
        self.D = coreD
        self.rD = {v: k for k, v in self.D.items()}
        self.trackers = []
        for xe in self.trackables:
            tracker = NodeTrackerDF(xe)
            self.trackers.append(tracker)
        self._dirty_ids = set()
        self._did_the_end = [False] * len(self.trackers)

    def reset(self, *which, **kw):
        force = q.getkw(kw, "force", default=False)
        if len(which) > 0:
            for w in which:
                self.trackers[w].reset()
                self._did_the_end[w] = False
        else:
            if not force and len(self._dirty_ids) > 0:
                self.reset(*list(self._dirty_ids))
                self._dirty_ids = set()
            else:
                for tracker in self.trackers:
                    tracker.reset()
                self._did_the_end = [False] * len(self.trackers)

    def get_valid_next(self, eid):
        tracker = self.trackers[eid]
        nvt = tracker._nvt      # with structure annotation
        if len(nvt) == 0:
            if self._did_the_end[eid] is True:
                # nvt = {u'<RARE>'}
                nvt = {'<MASK>'}           # <-- why was rare? loss handles -inf on mask now
            else:
                nvt = {"<END>"}
                self._did_the_end[eid] = True
                self._dirty_ids.add(eid)
        _nvt = set()
        for x in nvt:
            x = x.replace('*NC', '').replace('*LS', '')
            _nvt.add(x)
        nvt = [self.D[x] for x in _nvt]
        return nvt

    def update(self, eid, x, alt_x=None):
        tracker = self.trackers[eid]
        self._dirty_ids.add(eid)
        nvt = tracker._nvt
        if len(nvt) == 0:
            pass
        else:
            core = self.rD[x]
            suffix = ''
            if core not in "<QUERY> <SELECT> <WHERE> <COND> <VAL>".split():
                suffix += '*NC'
            if core in '<ENDVAL> <END> <QUERY>'.split():
                suffix += '*LS'
            else:       # check previous _nvt, if it occurred as *LS there, do *LS
                if core + suffix in nvt and not (core + suffix + '*LS' in nvt):
                    suffix += ''
                elif core + suffix + '*LS' in nvt and not (core + suffix in nvt):
                    suffix += '*LS'
                else:
                    suffix += ''
                    print("sum ting wong in sql tracker df !!!!!!!!!!!")
            x = core + suffix
            tracker.nxt(x)

    def is_terminated(self, eid):
        return self.trackers[eid].is_terminated() and self._did_the_end[eid] is True


def make_tracker_df(osm):
    tt = q.ticktock("tree tracker maker")
    tt.tick("making trees")
    trees = []
    for i in range(len(osm.matrix)):
        tree = SqlNode.parse_sql(osm[i])
        trees.append(tree)
    tracker = SqlGroupTrackerDF(trees, osm.D)
    tt.tock("trees made")
    return tracker


# region test
def tst_grouptracker():
    # TODO: test that every possible tree is correct tree and leads to correct sql
    ism, osm, csm, psm, splits, e2cn = load_matrices()
    devstart, teststart = splits

    tracker = make_tracker_df(osm)

    devstart = 74500

    for i in range(devstart, len(osm.matrix)):
        accs = set()
        numconds = len(re.findall("<COND>", tracker.trackables[i].pp()))
        numsamples = {0: 3, 1: 3, 2: 5, 3: 10, 4: 100, 5: 1000}[numconds]
        for j in range(numsamples):
            acc = ""
            tracker.reset()
            while True:
                if tracker.is_terminated(i):
                    break
                vnt = tracker.get_valid_next(i)
                sel = random.choice(vnt)
                acc += " " + tracker.rD[sel]
                tracker.update(i, sel)
            accs.add(acc)
            assert(SqlNode.parse_sql(acc).equals(tracker.trackables[i]))
            if not SqlNode.parse_sql(unidecode(acc)).equals(tracker.trackables[i]):
                print(acc)
                print(tracker.trackables[i].pptree())
                print(SqlNode.parse_sql(unidecode(acc)).pptree())
                raise q.SumTingWongException("trees not equal")
        assert(len(accs) > 0)
        print("number of unique linearizations for example {}: {} - {}".format(i, len(accs), numconds))

# endregion

# endregion

# endregion SQL TREES

# region DYNAMIC VECTORS
# region intro
from qelos_core.word import WordEmbBase, WordLinoutBase

class DynamicVecComputer(torch.nn.Module):  pass
class DynamicVecPreparer(torch.nn.Module):  pass
# endregion

# region dynamic word emb and word linout in general
class DynamicWordEmb(WordEmbBase):
    """ Dynamic Word Emb dynamically computes word embeddings on a per-example basis
        based on the given batch of word ids and batch of data.

        Computer can be a DynamicVecPreparer or a DynamicVecComputer.
        The .prepare() method must be called at the beginning of every batch
            with the per-example data batch used to compute the vectors.

        If a Preparer is used, vectors are prepared for every example in the batch.
        The Preparer receives the data, per example and must compute different sets of vectors for different examples.
        During forward, the prepared vectors are sliced from, on a per-example basis.

        If a Computer is used, .prepare() only stores the given data,
            which are passed to the computer during forward, together with the actual input.
        """

    def __init__(self, computer=None, worddic=None, **kw):
        super(DynamicWordEmb, self).__init__(worddic=worddic)
        self.computer = computer
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        self.maskid = maskid
        self.indim = max(worddic.values()) + 1
        self.saved_data = None

    def prepare(self, *xdata):
        if isinstance(self.computer, DynamicVecPreparer):
            ret, _ = self.computer.prepare(*xdata)
            self.saved_data = ret
        else:
            self.saved_data = xdata

    def forward(self, x):
        mask = None
        if self.maskid is not None:
            mask = x != self.maskid
        emb = self._forward(x)
        return emb, mask

    def _forward(self, x):
        if isinstance(self.computer, DynamicVecComputer):
            return self.computer(x, *self.saved_data)
        else:       # default implementation
            assert(isinstance(self.computer, DynamicVecPreparer))
            vecs = self.saved_data
            xdim = x.dim()
            if xdim == 1:
                x = x.unsqueeze(1)
            ret = vecs.gather(1, x.clone().unsqueeze(2).repeat(1, 1, vecs.size(2)))
            if xdim == 1:
                ret = ret.squeeze(1)
            return ret

    def __eq__(self, other):
        return self.computer == other.computer and self.maskid == other.maskid \
               and self.indim == other.indim and self.D == other.D


class DynamicWordLinout(WordLinoutBase):        # removed the logsoftmax in here
    """ Must be used with a DynamicVecPreparer (see DynamicWordEmb doc).
        As with DynamicWordEmb, the vectors used in this layer are different for every example.
        .prepare() must be called with the per-example data batch at the beginning of the batch.
    """
    def __init__(self, computer=None, worddic=None, **kw):
        super(DynamicWordLinout, self).__init__(worddic, **kw)
        self.computer = computer
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        self.maskid = maskid
        self.outdim = max(worddic.values()) + 1
        self.saved_data = None

    def prepare(self, *xdata):
        assert (isinstance(self.computer, DynamicVecPreparer))
        ret = self.computer.prepare(*xdata)
        self.saved_data = ret

    def forward(self, x, mask=None, _no_mask_log=False, **kw):
        ret, rmask = self._forward(x)
        if rmask is not None:
            if mask is None:
                mask = rmask
            else:
                mask = mask * rmask
        if mask is not None:
            if _no_mask_log is False:
                ret = ret + torch.log(mask.float())
            else:
                ret = ret * mask.float()
        return ret

    def _forward(self, x):
        assert (isinstance(self.computer, DynamicVecPreparer))
        vecs = self.saved_data
        rmask = None
        if len(vecs) == 2:
            vecs, rmask = vecs
        xshape = x.size()
        if len(xshape) == 2:
            x = x.unsqueeze(1)
        else:  # 3D -> unsqueeze rmask, will be repeated over seq dim of x
            if rmask is not None:
                rmask = rmask.unsqueeze(1)
        ret = torch.bmm(x, vecs.transpose(2, 1))
        if len(xshape) == 2:
            ret = ret.squeeze(1)
        return ret, rmask

    def __eq__(self, other):
        return self.computer == other.computer and self.maskid == other.maskid \
               and self.outdim == other.outdim and self.D == other.D
# endregion


# region dynamic vectors modules
class ColnameEncoder(torch.nn.Module):
    """ Encoder for column names.
        Uses one LSTM layer.
    """
    def __init__(self, dim, colbaseemb, nocolid=None, useskip=False):
        super(ColnameEncoder, self).__init__()
        self.emb = colbaseemb
        self.embdim = colbaseemb.vecdim
        self.dim = dim
        self.enc = torch.nn.LSTM(self.embdim, self.dim, 1, batch_first=True)
        self.nocolid = nocolid
        self.useskip = useskip      # add average of word vectors to bottom part of encoding

    def forward(self, x):
        """ input is (batsize, numcols, colnamelen)
            out is (batsize, numcols, dim)
        """
        # TODO: test
        rmask = None
        if self.nocolid is not None:
            rmask = x[:, :, 0] != self.nocolid
        xshape = x.size()
        flatx = x.contiguous().view(-1, x.size(-1))
        embx, mask = self.emb(flatx)
        c_0 = torch.zeros(1, flatx.size(0), self.dim, device=x.device)
        y_0 = c_0
        packedx, order = q.seq_pack(embx, mask)
        _y_t, (y_T, c_T) = self.enc(packedx, (y_0, c_0))
        y_T = y_T[0][order]
        # y_t, umask = q.seq_unpack(_y_t, order)
        ret = y_T.contiguous().view(x.size(0), x.size(1), y_T.size(-1))
        if self.useskip:
            embxx = embx.view(x.size(0), x.size(1), x.size(2), -1)
            maskx = mask.view(x.size(0), x.size(1), x.size(2)).float().unsqueeze(3)
            avg_emb = embxx * maskx
            avg_emb = avg_emb.sum(2)
            avg_emb = avg_emb / maskx.sum(2)
            if avg_emb.size(2) < ret.size(2):
                xtr_emb = torch.zeros(x.size(0), x.size(1), ret.size(2) - avg_emb.size(2)).to(embx.device)
                avg_emb = torch.cat([xtr_emb, avg_emb], 2)       # put embeddings at the end
            elif avg_emb.size(2) > ret.size(2):
                avg_emb = avg_emb[:, :, :ret.size(2)]
            else:
                pass
            ret = ret + avg_emb
        return ret, rmask

    def __eq__(self, other):
        return self.emb == other.emb and self.embdim == other.embdim and self.dim == other.dim \
               and self.enc == other.enc and self.nocolid == other.nocolid and self.useskip == other.useskip


class OutvecComputer(DynamicVecPreparer):
    """ This is a DynamicVecPreparer used for both output embeddings and output layer.
        To be created, needs:
         * syn_emb:         normal syntax embedder_syn_embs.shape
         * syn_trans:       normal syntax trans - if sliced with osm.D ids, return ids to use with syn_emb in .prepare()
         * inpbaseemb:      embedder for input words
         * inp_trans:       input words trans - if sliced with osm.D ids, return ids to use with inpbaseemb and inpmaps in .prepare()
         * colencoder:      encoder for column names
         * col_trans:       column names trans
         * worddic:         dictionary of output symbols (synids, colids, uwids) = osm.D
    """
    def __init__(self, syn_emb, inpbaseemb, colencoder, worddic,
                 syn_scatter, inp_scatter, col_scatter,
                 rare_gwids=None):
        super(OutvecComputer, self).__init__()
        self.syn_emb = syn_emb
        self.inp_emb = inpbaseemb
        self.col_enc = colencoder
        self.syn_scatter = q.val(syn_scatter).v
        self.inp_scatter = q.val(inp_scatter).v
        self.col_scatter = q.val(col_scatter).v
        self.D = worddic

        # initialize rare vec to rare vector from syn_emb
        # self.rare_vec = torch.nn.Parameter(syn_emb.embedding.weight[syn_emb.D["<RARE>"]].detach())
        self.rare_vec = torch.nn.Parameter(torch.randn(syn_emb.embedding.weight.size(1)))
        self.rare_gwids = rare_gwids

        if self.inp_emb.vecdim != self.syn_emb.vecdim:
            print("USING LIN ADAPTER in OUT")
            self.inpemb_trans = torch.nn.Linear(self.inp_emb.vecdim, self.syn_emb.vecdim, bias=False)
        else:
            self.inpemb_trans = None

    def __eq__(self, other):
        return self.syn_emb == other.syn_emb and self.inp_emb == other.inp_emb and self.col_enc == other.col_enc \
               and self.syn_scatter == other.syn_scatter and self.inp_scatter == other.inp_scatter \
               and self.col_scatter == other.col_scatter and self.D == other.D and self.rare_gwids == other.rare_gwids

    def prepare(self, inpmaps, colnames):
        """ inpmaps (batsize, num_uwids) contains mapping from uwids to gwids for every example = batch from gwids matrix
            colnames (batsize, numcols, colnamelen) contains colnames for every example
        """
        batsize = inpmaps.size(0)
        syn_ids = torch.arange(0, self.syn_scatter.size(0), device=inpmaps.device, dtype=torch.int64)
        inp_ids = torch.arange(0, self.inp_scatter.size(0), device=inpmaps.device, dtype=torch.int64)
        col_ids = torch.arange(0, self.col_scatter.size(0), device=inpmaps.device, dtype=torch.int64)

        syn_embs, syn_mask = self.syn_emb(syn_ids.unsqueeze(0).repeat(batsize, 1))
        if syn_mask is None:
            syn_mask = torch.ones_like(syn_embs[:, :, 0])
        embdim = syn_embs.size(2)

        inp_ids_gwid = torch.gather(inpmaps, 1, inp_ids.unsqueeze(0).repeat(batsize, 1))
        inp_embs, inp_mask = self.inp_emb(inp_ids_gwid)
        if self.inpemb_trans is not None:
            inp_embs = self.inpemb_trans(inp_embs)
        inp_embs = replace_rare_gwids_with_rare_vec(inp_embs, inp_ids_gwid, self.rare_gwids, self.rare_vec)

        col_encs, col_mask = self.col_enc(colnames)
        col_encs = col_encs * col_mask.float().unsqueeze(2)

        out_embs = torch.zeros(batsize, max(self.D.values()) + 1, embdim, device=inpmaps.device)
        out_mask = torch.zeros_like(out_embs[:, :, 0])

        out_embs.scatter_(1, self.inp_scatter.unsqueeze(0).unsqueeze(2).repeat(batsize, 1, embdim),
                          inp_embs)
        out_embs.scatter_(1, self.col_scatter.unsqueeze(0).unsqueeze(2).repeat(batsize, 1, embdim),
                          col_encs)
        out_embs.scatter_(1, self.syn_scatter.unsqueeze(0).unsqueeze(2).repeat(batsize, 1, embdim),
                          syn_embs)

        out_mask.scatter_(1, self.inp_scatter.unsqueeze(0).repeat(batsize, 1), inp_mask.float())
        out_mask.scatter_(1, self.col_scatter.unsqueeze(0).repeat(batsize, 1), col_mask.float())
        out_mask.scatter_(1, self.syn_scatter.unsqueeze(0).repeat(batsize, 1), syn_mask.float())

        return out_embs, out_mask


class PtrGenOut(DynamicWordLinout, q.AutoMaskedOut):
    def __init__(self, core, worddic=None, automasker=None):
        """
        Wraps PointerGeneratorOut, makes it prepareable wrt outveccomp
        :param core:            a PointerGeneratorOut with a OutVecComputer as gen_out module
        :param worddic:
        """
        super(PtrGenOut, self).__init__(computer=core.gen_out[0].computer, worddic=worddic, automasker=automasker)
        self.core = core

    def prepare(self, *xdata):
        self.core.gen_out[0].prepare(*xdata)

    def forward(self, x, alphas, ctx_inp, **kw):
        mask = None
        if self.automasker is not None:
            mask = self.automasker.get_out_mask()
        ret = self.core(x, alphas, ctx_inp, mask=mask)
        return ret


class MyAutoMasker(q.AutoMasker):
    def __init__(self, inpD, outD, ctxD=None, selectcolfirst=False, usesem=True, **kw):
        super(MyAutoMasker, self).__init__(inpD, outD, **kw)
        self.selectcolfirst = selectcolfirst
        self.flags = None
        self.inpseqs = None
        self.coltypes = None
        self.ctxD = ctxD
        self.RctxD = {v: k for k, v in ctxD.items()}
        self.usesem = usesem

    def reset(self):
        super(MyAutoMasker, self).reset()
        self.flags = None
        self.inpseqs = None
        self.coltypes = None

    def update_inpseq(self, inpseqs):
        """ (batsize, seqlen) ^ integer ids in inp voc (uwids)"""
        if self.test_only and self.training:
            pass
        else:
            self.inpseqs = []
            for i in range(len(inpseqs)):
                inpseq = list(inpseqs[i].cpu().detach().numpy())
                inpseq = [self.RctxD[inpseq_e] for inpseq_e in inpseq if inpseq_e != 0]
                followers = {k: set() for k in set(inpseq)}
                for f, t in zip(inpseq[:-1], inpseq[1:]):
                    followers[f].add(t)
                followers = {k: list(v) for k, v in followers.items()}
                self.inpseqs.append(followers)

    def update_coltypes(self, coltypeses):
        """ (batsize, numcols) ^ integer ids of column types "real" = 1, "text" = 2 """
        if self.test_only and self.training:
            pass
        else:
            self.coltypes = []
            td = {1: "real", 2: "text", 0: "<MASK>"}
            for i in range(len(coltypeses)):
                coltypes = list(coltypeses[i].cpu().detach().numpy())
                coltypes = [td[coltype] for coltype in coltypes if coltype != 0]
                coltypes = {"COL{}".format(k): v for k, v in zip(range(len(coltypes)), coltypes)}
                self.coltypes.append(coltypes)

    def get_out_tokens_for_history(self, i, hist):
        if self.test_only and self.training:
            return None
        else:
            if not hasattr(self, "flags") or self.flags is None:
                self.flags = {}

            if i not in self.flags:
                self.flags[i] = {"inselect": False}

            prev = hist[-1]

            def get_rets(k, coltype=None):
                ret = list([x for x in self.outD.keys() if k == x[:len(k)]])
                if self.usesem:
                    if coltype == "text":   # restrict aggs and ops
                        def text_col_filter_fun(x):
                            if re.match("OP\d+", x):
                                return x in ["OP0"]     # allowed ops after a "text" column
                            elif re.match("AGG\d+", x):
                                return x in ["AGG0", "AGG3"]    # allowed aggs after a "text" column
                            else:
                                return True
                        ret = list(filter(text_col_filter_fun, ret))
                return ret

            if prev == "<START>":
                ret = ["<QUERY>"]
            elif prev == "<QUERY>":
                ret = ["<SELECT>"]
            elif prev == "<SELECT>":
                self.flags[i]["inselect"] = True
                if self.selectcolfirst:
                    ret = get_rets("COL")
                else:
                    ret = get_rets("AGG")
            elif prev == "<WHERE>":
                self.flags[i]["inselect"] = False
                ret = ["<COND>"]
            elif prev == "<COND>":
                ret = get_rets("COL")
            elif re.match("COL\d+", prev):
                if self.flags[i]["inselect"] == True:
                    if self.selectcolfirst:
                        ret = get_rets("AGG", coltype=self.coltypes[i][prev])
                    else:
                        ret = ["<WHERE>", "<END>"]
                else:
                    ret = get_rets("OP", coltype=self.coltypes[i][prev])
            elif re.match("AGG\d+", prev):
                if self.selectcolfirst:
                    ret = ["<WHERE>", "<END>"]
                else:
                    ret = get_rets("COL")
            elif re.match("OP\d+", prev):
                ret = ["<VAL>"]
            elif prev == "<VAL>":
                ret = get_rets("UWID")
            elif re.match("UWID\d+", prev):
                if self.usesem:
                    ret = self.inpseqs[i][prev]
                else:
                    ret = get_rets("UWID")
                ret += ["<ENDVAL>"]
            elif prev == "<ENDVAL>":
                ret = ["<COND>", "<END>"]
            elif prev in "<END> <MASK>".split():
                ret = ["<MASK>"]
            else:
                raise q.SumTingWongException("token {} in example {} not covered".format(prev, i))
            return ret
# endregion


# region dynamic vector modules helper functions
def replace_rare_gwids_with_rare_vec(x, ids, rare_gwids, rare_vec):
    if rare_gwids is None:
        return x
    # get mask based on where rare_gwids occur in ids
    ids_np = ids.cpu().data.numpy()
    ids_mask_np = np.vectorize(lambda x: x not in rare_gwids)(ids_np).astype("uint8")       # ids_mask is one if NOT rare
    ids_mask = torch.tensor(ids_mask_np, dtype=torch.float32).to(x.device)
    # switch between vectors
    ret = rare_vec.unsqueeze(0).unsqueeze(1) * (1 - ids_mask.unsqueeze(2))
    ret = ret + x * ids_mask.unsqueeze(2)
    # TODO: test properly
    return ret


def build_subdics(osmD):
    # split dictionary for SQL syntax, col names and input tokens
    synD = {"<MASK>": 0}
    colD = {}
    inpD = {}
    # the _trans's below indicate what a osmD position is in terms of each of the three subdics
    syn_trans = q.val(np.zeros((len(osmD),), dtype="int64")).v
    inp_trans = q.val(np.zeros((len(osmD),), dtype="int64")).v
    col_trans = q.val(np.zeros((len(osmD),), dtype="int64")).v
    col_trans.data.fill_(-1)

    for k, v in osmD.items():
        m = re.match('(UWID|COL)(\d+)', k)
        if m:
            if m.group(1) == "UWID":
                if k not in inpD:
                    inpD[k] = int(m.group(2))
                    inp_trans.data[v] = inpD[k]
            elif m.group(1) == "COL":
                if k not in colD:
                    colD[k] = int(m.group(2))
                    col_trans.data[v] = colD[k]
        else:
            if k not in synD:
                synD[k] = len(synD)
                syn_trans.data[v] = synD[k]

    # scatters specify where in outdic to put the indexed local position
    syn_scatter = torch.ones(max(synD.values())+1, dtype=torch.int64) * 0
    inp_scatter = torch.ones(max(inpD.values())+1, dtype=torch.int64) * 0
    col_scatter = torch.ones(max(colD.values())+1, dtype=torch.int64) * 0

    for k, v in synD.items():
        syn_scatter[v] = osmD[k]
    for k, v in inpD.items():
        inp_scatter[v] = osmD[k]
    for k, v in colD.items():
        col_scatter[v] = osmD[k]

    # assert((inp_scatter > 0).all().item() == 1)
    # assert((col_scatter > 0).all().item() == 1)

    return synD, inpD, colD, syn_trans, inp_trans, col_trans, syn_scatter, inp_scatter, col_scatter


def make_out_vec_computer(dim, osmD, psmD, csmD, inpbaseemb=None, colbaseemb=None, colenc=None,
                          useglove=True, gdim=None, gfrac=0.1, useskip=False,
                          rare_gwids=None, nogloveforinp=False, no_maskzero=False):
    # base embedder for input tokens
    embdim = gdim if gdim is not None else dim
    if inpbaseemb is None:
        inpbaseemb = q.WordEmb(dim=embdim, worddic=psmD)
        if useglove and not nogloveforinp:
            inpbaseemb = q.PartiallyPretrainedWordEmb(dim=embdim, worddic=psmD, gradfracs=(1., gfrac))

    # base embedder for column names
    if colbaseemb is None:
        colbaseemb = q.WordEmb(embdim, worddic=csmD)
        if useglove:
            colbaseemb = q.PartiallyPretrainedWordEmb(dim=embdim, worddic=csmD, gradfracs=(1., gfrac))

    synD, inpD, colD, syn_trans, inp_trans, col_trans, syn_scatter, inp_scatter, col_scatter\
        = build_subdics(osmD)

    syn_emb = q.WordEmb(dim, worddic=synD, no_masking=no_maskzero)        # TODO: enable

    if colenc is None:
        colenc = ColnameEncoder(dim, colbaseemb, nocolid=csmD["nonecolumnnonecolumnnonecolumn"], useskip=useskip)

    # TODO: fix: backward breaks
    computer = OutvecComputer(syn_emb, inpbaseemb, colenc, osmD,
                              syn_scatter, inp_scatter, col_scatter,
                              rare_gwids=rare_gwids)

    # computer = OutVecComputer(syn_emb, syn_trans, inpbaseemb, inp_trans, colenc, col_trans, osmD,
    #                           rare_gwids=rare_gwids, scatters=[syn_scatter, inp_scatter, col_scatter])
    return computer, inpbaseemb, colbaseemb, colenc
# endregion


# region dynamic vector module creation functions
def make_inp_emb(dim, ismD, psmD, useglove=True, gdim=None, gfrac=0.1,
                 rare_gwids=None):
    """
    :param dim:         dimensionality of embeddings
    :param ism:         UWIDs for every example's input
    :param psm:         gwids -- mapping fromautomasker UWIDs from ism to words
    :param useglove:    whether to use glove --> partially pretrained wordembs
    :param gdim:        glove dim
    :param gfrac:       lr fraction for glove vectors
    :param rare_gwids:  rare
    :return:
    """
    embdim = gdim if gdim is not None else dim
    baseemb = q.WordEmb(dim=embdim, worddic=psmD)
    if useglove:
        baseemb = q.PartiallyPretrainedWordEmb(dim=embdim, worddic=psmD, gradfracs=(1., gfrac))

    class Computer(DynamicVecComputer):
        def __init__(self):
            super(Computer, self).__init__()
            self.baseemb = baseemb
            self.rare_gwids = rare_gwids
            # self.rare_vec = torch.nn.Parameter(baseemb.embedding.weight[baseemb.D["<RARE>"]].detach())
            self.rare_vec = torch.nn.Parameter(torch.randn(baseemb.embedding.weight.size(1)))
            if embdim != dim:
                print("USING LIN ADAPTER")
                self.trans = torch.nn.Linear(embdim, dim, bias=False)
            else:
                self.trans = None

        def forward(self, x, data):
            transids = torch.gather(data, 1, x)         # transids are in gwids
            # _pp = psm.pp(transids[:5].cpu().data.numpy())
            _embs, mask = self.baseemb(transids)        # baseemb embedds gwids
            if self.trans is not None:
                _embs = self.trans(_embs)
            _embs = replace_rare_gwids_with_rare_vec(_embs, transids, self.rare_gwids, self.rare_vec)
            return _embs

    emb = DynamicWordEmb(computer=Computer(), worddic=ismD)
    return emb, baseemb


def make_out_emb(dim, osmD, psmD, csmD, inpbaseemb=None, colbaseemb=None,
                 useglove=True, gdim=None, gfrac=0.1, colenc=None,
                 rare_gwids=None, useskip=False):
    print("MAKING OUT EMB")
    comp, inpbaseemb, colbaseemb, colenc \
        = make_out_vec_computer(dim, osmD, psmD, csmD, inpbaseemb=inpbaseemb, colbaseemb=colbaseemb,
                                colenc=colenc, useglove=useglove, gdim=gdim, gfrac=gfrac,
                                rare_gwids=rare_gwids, useskip=useskip)
    return DynamicWordEmb(computer=comp, worddic=osmD), inpbaseemb, colbaseemb, colenc


def make_out_lin(dim, ismD, osmD, psmD, csmD, inpbaseemb=None, colbaseemb=None,
                 useglove=True, gdim=None, gfrac=0.1, colenc=None, nocopy=False,
                 rare_gwids=None, automasker=None, ptrgenmode="sepsum", useoffset=False, useskip=False):
    print("MAKING OUT LIN")
    comp, inpbaseemb, colbaseemb, colenc \
        = make_out_vec_computer(dim, osmD, psmD, csmD, inpbaseemb=inpbaseemb, colbaseemb=colbaseemb,
                                colenc=colenc, useglove=useglove, gdim=gdim, gfrac=gfrac,
                                rare_gwids=rare_gwids, nogloveforinp=False, no_maskzero=True,
                                useskip=useskip)

    out = torch.nn.Sequential(DynamicWordLinout(comp, osmD),)

    if not nocopy:
        gen_zero_set = set(ismD.keys()) - set(["<MASK>"])

        if ptrgenmode == "sepsum":
            switcher = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.Tanh(),
                torch.nn.Linear(dim, 1),
                torch.nn.Sigmoid(),
            )

            ptrgenout = q.PointerGeneratorOutSeparate(osmD, switcher, out, inpdic=ismD,
                                                      gen_zero=gen_zero_set,
                                                      gen_outD=osmD)
        elif ptrgenmode == "sharemax":
            if useoffset:
                ptroffsetter = torch.nn.Sequential(
                    torch.nn.Linear(dim, dim),
                    torch.nn.Tanh(),
                    torch.nn.Linear(dim, 1),
                )
            else:
                ptroffsetter = None
            ptrgenout = q.PointerGeneratorOutSharedMax(osmD, out, ptr_offsetter=ptroffsetter,
                                                       inpdic=ismD, gen_zero=gen_zero_set, gen_outD=osmD)
        else:
            raise q.SumTingWongException("unknown ptrgenmode: {}".format(ptrgenmode))
        out = PtrGenOut(ptrgenout, worddic=osmD, automasker=automasker)
        # DONE: use PointerGeneratorOut here
        # 1. create generation block (create dictionaries for pointergenout first)
        #           generator can be DynamicWordLinout with a normal softmax
        #           then can just give ismD and osmD and generate a gen_zero set to override generator probs for input tokens
        # 2. create gen vs ptr switcher module
        # 3. create pointergenout
        # DONE: how about mask on unused colids?? --> OutVecComputer returns a mask when preparing
        # mask on unused inputs: should be handled normally by default pointergenout
        # out = BFOL(computer=comp, worddic=osmD, ismD=ismD, inp_trans=inp_trans, nocopy=nocopy)
    return out, inpbaseemb, colbaseemb, colenc

# endregion


# endregion

# region MAIN SCRIPTS
# region main scripts helper functions
def get_rare_stats(trainism, traingwids, gwidsD, gdic, rarefreq=2):
    """  get rare gwids, missing gwids, ignoring glove (glove should be taken into account in the modules) """
    rD = {v: k for k, v in gwidsD.items()}
    # count all unique ids in traingwids
    uniquegwids, gwid_counts = np.unique(traingwids, return_counts=True)
    numunique = len(uniquegwids)
    rare_gwids = gwid_counts <= rarefreq
    number_rare = np.sum(rare_gwids.astype("int32"))
    print("{}/{} gwids with freq <= {} (counted in train)".format(number_rare, numunique, rarefreq))
    unique_nonrare_ids = uniquegwids * (~rare_gwids).astype("int32")
    unique_nonrare_ids = set(unique_nonrare_ids)
    unique_nonrare_words = set([rD[unrid] for unrid in unique_nonrare_ids])
    rare_words = set(gwidsD.keys()) - unique_nonrare_words - set(gdic.keys())
    rare_gwids_after_glove = set([gwidsD[rare_word] for rare_word in rare_words])
    print("{}/{} gwids with freq <= {} (counted in train) and not in used glove".format(len(rare_gwids_after_glove), numunique, rarefreq))
    return rare_gwids_after_glove


def do_rare_in_colnames(cnsm, traincolids, gdic, rarefreq=3, replace=False):
    """ cnsm: (#cols, colnamelen) -- traincolids: (numex, #cols) """
    # print(np.sum(cnsm._matrix == cnsm.D["<RARE>"]))
    tt = q.ticktock("rare column names")
    tt.tick("computing counts")
    rD = {v: k for k, v in cnsm.D.items()}
    # print("{} words in cnsm.D".format(len(cnsm.D)))
    wordcounts = {w: 0 for w in cnsm.D.keys()}

    uniquecolids, colidcounts = np.unique(traincolids, return_counts=True)
    for i in range(len(uniquecolids)):
        for j in range(len(cnsm.matrix[uniquecolids[i]])):
            word = rD[cnsm.matrix[uniquecolids[i], j]]
            wordcounts[word] += colidcounts[i]

    rarewords = set()
    for k, v in wordcounts.items():
        if v <= rarefreq:
            rarewords.add(k)
    # print("nonecolumnnonecolumnnonecolumn" in rarewords)
    rarewords -= {"nonecolumnnonecolumnnonecolumn",}
    print("{} rare words (rarefreq {}) out of {} unique words in col names"
          .format(len(rarewords), rarefreq, len(wordcounts)))
    rarewords_notinglove = rarewords - set(gdic.keys())
    print("{} rare words (rarefreq {}) not in glove, out of {} unique words in col names"
          .format(len(rarewords_notinglove), rarefreq, len(wordcounts)))
    tt.tock("counts computed")
    if replace:
        cnsm._matrix = np.vectorize(lambda x: x if rD[x] not in rarewords_notinglove else cnsm.D["<RARE>"])(cnsm.matrix)
        tt.msg("total rare words replaced with rare id: {}".format(np.sum(cnsm.matrix == cnsm.D["<RARE>"])))
    return cnsm


def reorder_tf(osm, reordermode="no"):
    tt = q.ticktock("reorderer")
    if reordermode == "no":
        return osm
    elif reordermode in ("reverse", "arbitrary"):
        tt.tick("reordering")
        for i in range(len(osm.matrix)):
            lin = osm[i]
            tree = SqlNode.parse_sql(lin)
            if reordermode == "reverse":
                # get WHERE node
                wherenodes = list(get_children_by_name(tree, "<WHERE>"))
                assert(len(wherenodes) < 2)
                if len(wherenodes) == 1 and len(wherenodes[0].children) > 1:
                    wherenodes[0].children = wherenodes[0].children[::-1]
            relin = "<START> " + tree.pp_sql(arbitrary=reordermode == "arbitrary") + " <END>"
            relinids = [osm.D[x] for x in relin.split()]
            osm.matrix[i, :] = 0
            osm.matrix[i, :len(relinids)] = relinids
            newtree = SqlNode.parse_sql(osm[i])
            assert(newtree.equals(tree))
        tt.tock("reordered")
        return osm


def reorder_select(osm):
    tt = q.ticktock("select-reorder")
    tt.tick("putting col before agg in select")
    for i in range(len(osm.matrix)):
        lin = osm[i]
        tree = SqlNode.parse_sql(lin)
        selectnode = list(get_children_by_name(tree, "<SELECT>"))
        assert(len(selectnode) == 1)
        selectnode = selectnode[0]
        selectchildren = selectnode.children[::-1]
        for j, selectchild in enumerate(selectchildren):
            selectchild.order = j
        selectnode.children = selectchildren
        relin = "<START> " + tree.pp_sql() + " <END>"
        relinids = [osm.D[x] for x in relin.split()]
        osm.matrix[i, :] = 0
        osm.matrix[i, :len(relinids)] = relinids
        newtree = SqlNode.parse_sql(osm[i])
        assert(newtree.equals(tree))

    tt.tock("reordered")
    return osm


def get_output(model, data, origquestions, batsize=100, inp_bt=None, device=torch.device("cpu"),
               rev_osm_D=None, rev_gwids_D=None):
    """ takes a model (must be freerunning !!!) """
    gwids = data[2]
    # TODO: make sure q.eval() doesn't feed anything wrong or doesn't forget to reset things
    dataloader = q.dataload(*data, batch_size=batsize, shuffle=False)
    predictions = q.eval(model).on(dataloader) \
        .set_batch_transformer(inp_bt).device(device).run()
    _, predictions = predictions.max(2)
    predictions = predictions.cpu().data.numpy()
    rawlines = []
    sqls = []
    for i in range(len(gwids)):
        rawline = reconstruct_query(predictions[i], gwids[i], rev_osm_D, rev_gwids_D)
        rawline = rawline.replace("<START>", "").strip()
        rawline = rawline.split("<END>")[0].strip()
        rawlines.append(rawline)
        sqljson = querylin2json(rawline, origquestions[i])
        sqls.append(sqljson)
    return rawlines, sqls


def compute_sql_acc(pred_sql, gold_sql):
    sql_acc = 0.
    sql_acc_norm = 1e-6
    for pred_sql_i, gold_sql_i in zip(pred_sql, gold_sql):
        sql_acc_norm += 1
        sql_acc += 1. if same_sql_json(gold_sql_i, pred_sql_i) else 0.
    return sql_acc / sql_acc_norm


def compute_seq_acc(pred_sql, gold_sql):
    sql_acc = 0.
    sql_acc_norm = 1e-6
    for pred_sql_i, gold_sql_i in zip(pred_sql, gold_sql):
        sql_acc_norm += 1
        sql_acc += 1. if same_lin_json(gold_sql_i, pred_sql_i) else 0.
    return sql_acc / sql_acc_norm


def evaluate_model(m, devdata, testdata, rev_osm_D, rev_gwids_D,
                   inp_bt=None, batsize=100, device=None, savedir=None, test=False):
    def save_lines(lines, fname):
        with codecs.open(savedir + '/' + fname, "w", encoding="utf-8") as f:
            for lin in lines:
                f.write("{}\n".format(lin))

    # dev predictions
    devquestions = load_jsonls(DATA_PATH + "dev.jsonl", questionsonly=True)
    devsqls = load_jsonls(DATA_PATH + "dev.jsonl", sqlsonly=True)

    if test:
        devquestions = load_jsonls(DATA_PATH + "train.jsonl", questionsonly=True)[200:250]
        devsqls = load_jsonls(DATA_PATH + "train.jsonl", sqlsonly=True)[200:250]

    pred_devlines, pred_devsqls = get_output(m, devdata, devquestions,
                                             batsize=batsize, inp_bt=inp_bt, device=device,
                                             rev_osm_D=rev_osm_D, rev_gwids_D=rev_gwids_D)
    if savedir is not None:
        save_lines(pred_devlines, "dev_pred.lines")
        save_lines([json.dumps(x) for x in pred_devsqls], "dev_pred.jsonl")

    dev_sql_acc = compute_sql_acc(pred_devsqls, devsqls)
    print("DEV SQL ACC: {}".format(dev_sql_acc))

    # test predictions
    testquestions = load_jsonls(DATA_PATH + "test.jsonl", questionsonly=True)
    testsqls = load_jsonls(DATA_PATH + "test.jsonl", sqlsonly=True)
    pred_testlines, pred_testsqls = get_output(m, testdata, testquestions,
                                               batsize=batsize, inp_bt=inp_bt, device=device,
                                               rev_osm_D=rev_osm_D, rev_gwids_D=rev_gwids_D)
    if savedir is not None:
        save_lines(pred_testlines, "test_pred.lines")
        save_lines([json.dumps(x) for x in pred_testsqls], "test_pred.jsonl")

    test_sql_acc = compute_sql_acc(pred_testsqls, testsqls)
    print("TEST SQL ACC: {}".format(test_sql_acc))
    return dev_sql_acc, test_sql_acc


def load_pred_jsonl(p):
    lines = open(p).readlines()
    lines = [json.loads(line) for line in lines]
    return lines


def get_accuracies(p, verbose=False):
    """ p is where experiment outputs are at"""
    if verbose:
        print(p)
    devsqls = load_jsonls(DATA_PATH + "dev.jsonl", sqlsonly=True)
    pred_devsqls = load_pred_jsonl(os.path.join(p, "dev_pred.jsonl"))
    dev_seq_acc = compute_seq_acc(pred_devsqls, devsqls)
    dev_sql_acc = compute_sql_acc(pred_devsqls, devsqls)
    if verbose:
        print("\tDEV SEQ ACC: {}".format(dev_seq_acc))
        print("\tDEV SQL ACC: {}".format(dev_sql_acc))

    testsqls = load_jsonls(DATA_PATH + "test.jsonl", sqlsonly=True)
    pred_testsqls = load_pred_jsonl(os.path.join(p, "test_pred.jsonl"))
    test_seq_acc = compute_seq_acc(pred_testsqls, testsqls)
    test_sql_acc = compute_sql_acc(pred_testsqls, testsqls)
    if verbose:
        print("\tTEST SEQ ACC: {}".format(test_seq_acc))
        print("\tTEST SQL ACC: {}".format(test_sql_acc))

    return dev_seq_acc, dev_sql_acc, test_seq_acc, test_sql_acc


# _ = get_avg_accs_of(".+s2s_new.+", completed=True, epochs=lambda x: x >15, selectcolfirst=True, userules="test", labelsmoothing=0.2, useskip=True, ptrgenmode="sharemax", synonly=True, reorder="no", test=False)
def get_avg_accs_of(*args, **kw):
    """ signature is forward to q.log.find_experiments(*args, **kw) to find matching experiments
        get_accuracies() is run for every found experiment and the average is returned """
    experiment_dirs = list(q.find_experiments(*args, **kw))
    accses = [[] for i in range(4)]
    for experiment_dir in experiment_dirs:
        accs = get_accuracies(experiment_dir, verbose=True)
        for acc, accse in zip(accs, accses):
            accse.append(acc * 100.)
    print("Average accs for {} selected experiments:".format(len(accses[0])))
    print("  DEV SEQ ACC: {:.2f}, std={:.2f}".format(np.mean(accses[0]), np.std(accses[0])))
    print("  DEV SQL ACC: {:.2f}, std={:.2f}".format(np.mean(accses[1]), np.std(accses[1])))
    print("  TEST SEQ ACC: {:.2f}, std={:.2f}".format(np.mean(accses[2]), np.std(accses[2])))
    print("  TEST SQL ACC: {:.2f}, std={:.2f}".format(np.mean(accses[3]), np.std(accses[3])))
    return accses


# region test
def tst_reconstruct_save_reload_and_eval():
    """ load matrices, get test set, reconstruct using get_output()'s logic, save, read, test """
    tt = q.ticktock("testsave")
    ism, osm, cnsm, gwids, splits, e2cn = load_matrices()
    rev_osm_D = {v: k for k, v in osm.D.items()}
    rev_gwids_D = {v: k for k, v in gwids.D.items()}
    _, teststart = splits
    testquestions = load_jsonls(DATA_PATH+"test.jsonl", questionsonly=True)
    testsqls = load_jsonls(DATA_PATH+"test.jsonl", sqlsonly=True)

    tt.tick("reconstructing test gold...")
    ism, osm, gwids = ism.matrix[teststart:], osm.matrix[teststart:], gwids.matrix[teststart:]
    rawlines, sqls = [], []
    for i in range(len(gwids)):
        rawline = reconstruct_query(osm[i], gwids[i], rev_osm_D, rev_gwids_D)
        rawline = rawline.replace("<START>", "").replace("<END>", "").strip()
        rawlines.append(rawline)
        sqljson = querylin2json(rawline, testquestions[i])
        sqls.append(sqljson)
    tt.tock("reconstructed")

    tt.tick("saving test gold...")
    # saving without codecs doesn't work
    with codecs.open("testsave.lines", "w", encoding="utf-8") as f:
        for line in rawlines:
            f.write("{}\n".format(line))

    with codecs.open("testsave.sqls", "w", encoding="utf-8") as f:
        for sql in sqls:
            f.write("{}\n".format(json.dumps(sql)))
    tt.tock("saved")

    tt.tick("reloading saved...")
    reloaded_lines = []
    with codecs.open("testsave.lines", encoding="utf-8") as f:
    # with open("testsave.lines") as f:
        for line in f:
            reloaded_lines.append(line.strip())

    reloaded_sqls = []
    # with codecs.open("testsave.sqls", encoding="utf-8") as f:
    with open("testsave.sqls") as f:
        for line in f:
            sql = json.loads(line)
            reloaded_sqls.append(sql)
    tt.tock("reloaded saved")

    for rawline, reloadedline in zip(rawlines, reloaded_lines):
        if not rawline == reloadedline:
            print(u"FAILED: '{}' \n - '{}'".format(rawline, reloadedline))
        assert(rawline == reloadedline)

    failures = 0
    for testsql, reloaded_sql in zip(testsqls, reloaded_sqls):
        if not same_sql_json(testsql, reloaded_sql):
            print("FAILED: {} \n - {} ".format(testsql, reloaded_sql))
            failures += 1
        # assert(same_sql_json(testsql, reloaded_sql))
    assert(failures == 1)
    print("only one failure")

    print(len(reloaded_lines))
# endregion

# endregion


# SHAREMAX runs:
# best one: python wikisql_seq2seq_tf_df.py -gdim 300 -dim 600 -epochs 30 -dorare -userules "test" -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 0 -useskip
# - sem rules:          python wikisql_seq2seq_tf_df.py -gdim 300 -dim 600 -epochs 30 -dorare -userules "test" -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 0 -useskip -synonly
# - rules:              python wikisql_seq2seq_tf_df.py -gdim 300 -dim 600 -epochs 30 -dorare -userules "no" -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 0 -useskip
# / rules in train:     python wikisql_seq2seq_tf_df.py -gdim 300 -dim 600 -epochs 30 -dorare -userules "both" -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 1 -useskip
# - label smoothing:    python wikisql_seq2seq_tf_df.py -gdim 300 -dim 600 -epochs 30 -dorare -userules "test" -selectcolfirst -labelsmoothing 0 -cuda -gpu 1 -useskip
# - skip:               python wikisql_seq2seq_tf_df.py -gdim 300 -dim 600 -epochs 30 -dorare -userules "test" -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 0

# SEPSUM runs:
# best one: python wikisql_seq2seq_tf_df.py -gdim 300 -dim 600 -epochs 30 -dorare -userules "test" -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 0 -useskip -ptrgenmode sepsum
# - rules:              python wikisql_seq2seq_tf_df.py -gdim 300 -dim 600 -epochs 30 -dorare -userules "no" -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 0 -useskip -ptrgenmode sepsum
# / rules in train:     python wikisql_seq2seq_tf_df.py -gdim 300 -dim 600 -epochs 30 -dorare -userules "both" -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 1 -useskip -ptrgenmode sepsum
# - label smoothing:    python wikisql_seq2seq_tf_df.py -gdim 300 -dim 600 -epochs 30 -dorare -userules "test" -selectcolfirst -labelsmoothing 0 -cuda -gpu 1 -useskip -ptrgenmode sepsum
# - skip:               python wikisql_seq2seq_tf_df.py -gdim 300 -dim 600 -epochs 30 -dorare -userules "test" -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 0 -ptrgenmode sepsum

# order runs
# python wikisql_seq2seq_tf_df.py -gdim 300 -dim 600 -epochs 20 -dorare -userules "test" -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 0 -useskip -reorder reverse
# python wikisql_seq2seq_tf_df.py -gdim 300 -dim 600 -epochs 20 -dorare -userules "test" -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 0 -useskip -reorder arbitrary
# python wikisql_seq2seq_oracle_df.py -gdim 300 -dim 600 -epochs 20 -dorare -userules test -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 0 -useskip -oraclemode zerocost
# python wikisql_seq2seq_oracle_df.py -gdim 300 -dim 600 -epochs 20 -dorare -userules test -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 0 -useskip -oraclemode sample

# uniform pretrain runs
# python wikisql_seq2seq_oracle_df.py -gdim 300 -dim 600 -epochs 20 -dorare -userules test -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 0 -useskip -oraclemode zerocost -uniformpretrain 3
# python wikisql_seq2seq_oracle_df.py -gdim 300 -dim 600 -epochs 20 -dorare -userules test -selectcolfirst -labelsmoothing 0.2 -cuda -gpu 1 -useskip -oraclemode sample -uniformpretrain 3
def run_seq2seq_tf(lr=0.001, batsize=100, epochs=50,
                   inpembdim=50, outembdim=50, innerdim=100, numlayers=2, dim=-1, gdim=-1,
                   dropout=0.2, rdropout=0.1, edropout=0., idropout=0.2, irdropout=0.1, dropouts=-1., rdropouts=-1., alldropouts=-1.,
                   wreg=1e-14, gradnorm=5., useglove=True, gfrac=0.,
                   cuda=False, gpu=0, tag="none", ablatecopy=False, test=False,
                   tieembeddings=False, dorare=False, reorder="no", selectcolfirst=False,
                   userules="no", ptrgenmode="sharemax", labelsmoothing=0., attmode="dot",
                   useoffset=False, smoothmix=0., coveragepenalty=0., useslotptr=False, useskip=False, synonly=False):
                    # userules: "no", "test", "both"
                    # reorder: "no", "reverse", "arbitrary"
                    # ptrgenmode: "sepsum" or "sharemax"
    # region init
    if alldropouts > 0.:
        dropouts, rdropouts = alldropouts, alldropouts
    if dropouts > 0.:
        dropout, idropout = dropouts, dropouts
    if rdropouts > 0.:
        rdropout, irdropout = rdropouts, rdropouts
    settings = locals().copy()
    logger = q.Logger(prefix="wikisql_s2s_new")
    logger.save_settings(**settings)
    logger.update_settings(completed=False)
    print("LOGGER PATH: {}".format(logger.p))
    logger.update_settings(version="1")

    model_save_path = os.path.join(logger.p, "model")

    print("Seq2Seq + TF (new)")
    print("PyTorch initial seed: {}".format(torch.initial_seed()))

    device = torch.device("cpu")
    if cuda:    device = torch.device("cuda", gpu)

    tt = q.ticktock("script")
    # endregion

    # region dimensions
    gdim = None if gdim < 0 else gdim

    if dim > 0:
        innerdim = dim              # total dimension of encoding
        inpembdim = dim // 2        # dimension of input embedding
        outembdim = dim // 2        # dimension of output embedding
        if gdim is not None:
            inpembdim = gdim
            outembdim = gdim

    outlindim = innerdim * 2    # dimension of output layer - twice the encoding dim because cat of enc and state

    encdim = innerdim // 2          # half the dimension because bidirectional encoder doubles it back
    encdims = [inpembdim] + [encdim] * numlayers    # encoder's layers' dimensions
    decdims = [outembdim] + [innerdim] * numlayers  # decoder's layers' dimensions
    # endregion

    # region data
    ism, osm, cnsm, gwids, splits, e2cn = load_matrices()
    column_types = load_coltypes()      # 2 = "text", 1 = "real"
    assert(np.allclose((column_types > 0), (e2cn > 0)))
    # ism: input in terms of UWIDs --> use gwids for actual words
                    #(UWID-X --> get X'ths word in gwids for that example;
                    # UWID-X: X starts from 1, so first column of gwids is 0 (mask))
    # osm: output in terms of SQL words and UWIDS (use gwids to map uwids)
    # cnsm: column name sm, keeps own dictionary of words
    # gwids: unique words in the input of the example, first element is 0 (mask) since UWIDS start from 1
    # splits: (devstart, teststart) -- where to split in train/valid/test
    # e2cn: maps eid to rows in cnsm
    gwids._matrix = gwids.matrix * (gwids.matrix != gwids.D["<RARE>"])
    devstart, teststart = splits
    eids = np.arange(0, len(ism), dtype="int64")

    osm = reorder_tf(osm, reordermode=reorder)
    if selectcolfirst:
        osm = reorder_select(osm)
    # q.embed()
    # splits
    if test:    devstart, teststart, batsize = 200, 250, 50
    datamats = [ism.matrix, osm.matrix, gwids.matrix, e2cn, column_types]
    traindata = [datamat[:devstart] for datamat in datamats]
    devdata = [datamat[devstart:teststart] for datamat in datamats]
    testdata = [datamat[teststart:] for datamat in datamats]

    rev_osm_D = {v: k for k, v in osm.D.items()}
    rev_gwids_D = {v: k for k, v in gwids.D.items()}

    gdic = q.PretrainedWordEmb(gdim).D
    rare_gwids_after_glove = get_rare_stats(traindata[0], traindata[2], gwids.D, gdic)
    print("{} doing rare".format("NOT" if not dorare else ""))
    if not dorare:
        rare_gwids_after_glove = None
    cnsm = do_rare_in_colnames(cnsm, traindata[-1], gdic, replace=dorare)
    # endregion

    # region submodules
    def create_submodules():
        _inpemb, inpbaseemb = make_inp_emb(inpembdim, ism.D, gwids.D, useglove=useglove, gdim=gdim, gfrac=gfrac,
                                           rare_gwids=rare_gwids_after_glove)
        _outemb, inpbaseemb, colbaseemb, _ = make_out_emb(outembdim, osm.D, gwids.D, cnsm.D, gdim=gdim,
                                              inpbaseemb=inpbaseemb, useglove=useglove, gfrac=gfrac,
                                              rare_gwids=rare_gwids_after_glove, useskip=useskip)
        if not tieembeddings:
            inpbaseemb, colbaseemb = None, None

        automasker = None
        if userules != "no":
            automasker = MyAutoMasker(osm.D, osm.D, ctxD=ism.D, selectcolfirst=selectcolfirst, usesem=not synonly)
            automasker.test_only = userules == "test"
        _outlin, inpbaseemb, colbaseemb, colenc = make_out_lin(outlindim, ism.D, osm.D, gwids.D, cnsm.D,
                                                              useglove=useglove, gdim=gdim, gfrac=gfrac,
                                                              inpbaseemb=inpbaseemb, colbaseemb=colbaseemb,
                                                              colenc=None, nocopy=ablatecopy, rare_gwids=rare_gwids_after_glove,
                                                               automasker=automasker, useskip=useskip,
                                                               ptrgenmode=ptrgenmode, useoffset=useoffset)

        _encoder = q.FastestLSTMEncoder(*encdims, dropout_in=idropout, dropout_rec=irdropout, bidir=True)
        return _inpemb, _outemb, _outlin, _encoder
    # inpemb, outemb, outlin, encoder = create_submodules()
    # endregion

    # region encoder-decoder definition
    class EncDec(torch.nn.Module):
        def __init__(self, _inpemb, _outemb, _outlin, _encoder, dec):
            super(EncDec, self).__init__()
            self.inpemb, self.outemb, self.outlin, self.encoder, self.decoder \
                = _inpemb, _outemb, _outlin, _encoder, dec

        def forward(self, inpseq, outseq, inpseqmaps, colnames, coltypes):
            # encoding
            self.inpemb.prepare(inpseqmaps)
            _inpembs, _inpmask = self.inpemb(inpseq)
            _inpenc = self.encoder(_inpembs, mask=_inpmask)
            inpmask = _inpmask[:, :_inpenc.size(1)]
            inpenc = q.intercat(_inpenc.chunk(2, -1), -1)       # TODO: do we need intercat?
            ctx = inpenc    # old normalpointer mode
            if useskip:
                ctxadd = torch.zeros(ctx.size(0), ctx.size(1), ctx.size(2) - _inpembs.size(2)).to(ctx.device)
                ctxadd = torch.cat([ctxadd, _inpembs[:, :ctx.size(1)]], 2)
                ctx = ctx + ctxadd

            # decoding
            self.outemb.prepare(inpseqmaps, colnames)
            self.outlin.prepare(inpseqmaps, colnames)

            if self.outlin.automasker is not None:
                self.outlin.automasker.update_inpseq(inpseq)
                self.outlin.automasker.update_coltypes(coltypes)

            decoding = self.decoder(outseq, ctx=ctx, ctx_mask=inpmask, ctx_inp=inpseq,
                                    maxtime=osm.matrix.shape[1]-1)
            # TODO: why -1 in maxtime?
            # --? maybe because that's max we need to do, given that gold seqs are -1 in len

            return decoding

        def __eq__(self, other):
            return self.inpemb == other.inpemb and self.outemb == other.outemb \
                   and self.outlin == other.outlin and self.encoder == other.encoder \
                   and self.decoder == other.decoder

    class EncDecSlotPtr(EncDec):
        """ Slot ptr for predicting select clause. Where clause the same"""
        def __init__(self, _inpemb, _outemb, _outlin, _encoder, dec):
            super(EncDecSlotPtr, self).__init__(_inpemb, _outemb, _outlin, _encoder, dec)
            self.slot_ptr_addr_lin = torch.nn.Linear(innerdim, 2, bias=False)
            self.slot_ptr_sm = torch.nn.Softmax(1)
            self.has_cond_pred = torch.nn.Sequential(torch.nn.Linear(innerdim, innerdim//2),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Linear(innerdim//2, 1),
                                                     torch.nn.Sigmoid(),)

        def forward(self, inpseq, outseq, inpseqmaps, colnames, coltypes):
            # encoding
            self.inpemb.prepare(inpseqmaps)
            _inpembs, _inpmask = self.inpemb(inpseq)
            _inpenc = self.encoder(_inpembs, mask=_inpmask)
            inpmask = _inpmask[:, :_inpenc.size(1)]
            final_ctx = self.encoder.y_n[-1]
            final_ctx = torch.cat([final_ctx[:, 0], final_ctx[:, 1]], 1)
            inpenc = q.intercat(_inpenc.chunk(2, -1), -1)       # TODO: do we need intercat?
            ctx = inpenc    # old normalpointer mode

            # decoding
            self.outemb.prepare(inpseqmaps, colnames)
            self.outlin.prepare(inpseqmaps, colnames)

            if self.outlin.automasker is not None:
                self.outlin.automasker.update_inpseq(inpseq)
                self.outlin.automasker.update_coltypes(coltypes)

            # region predicting separately
            # region slot ptr
            slot_ptr_scores = self.slot_ptr_addr_lin(ctx)
            slot_ptr_scores = slot_ptr_scores + torch.log(inpmask.unsqueeze(2).float())
            slot_ptr_addrs = self.slot_ptr_sm(slot_ptr_scores)
            slot_ptr_one = (ctx * slot_ptr_addrs[:, :, 0:1]).sum(1)
            slot_ptr_two = (ctx * slot_ptr_addrs[:, :, 1:2]).sum(1)
            slot_ptr_one = torch.cat([self.slot_ptr_addr_lin.weight[0, :].unsqueeze(0).repeat(inpseq.size(0), 1),
                                      slot_ptr_one], 1)
            slot_ptr_two = torch.cat([self.slot_ptr_addr_lin.weight[1, :].unsqueeze(0).repeat(inpseq.size(0), 1),
                                      slot_ptr_two], 1)
            # endregion

            # region prefix probs
            prefix = ["<START>", "<QUERY>", "<SELECT>", "<WHERE>", "<END>"]
            prefix = [osm.D[p] for p in prefix]
            prefix = torch.tensor(prefix).to(inpseq.device)
            prefix_probs = torch.zeros(inpseq.size(0), prefix.size(0), max(osm.D.values())+1).to(inpseq.device)
            prefix_probs.scatter_(2, prefix.unsqueeze(0).repeat(inpseq.size(0), 1).unsqueeze(2), 1.)
            ones = torch.ones(inpseq.size(0)).to(inpseq.device)
            # endregion

            # region select clause probs
            if self.outlin.automasker is not None:
                self.outlin.automasker.update(ones.long() * osm.D["<SELECT>"])

            select_arg_one_probs = self.outlin(slot_ptr_one, None, None)
            if self.outlin.automasker is not None:
                self.outlin.automasker.update(select_arg_one_probs.max(1)[1])
            select_arg_two_probs = self.outlin(slot_ptr_two, None, None)
            # endregion

            # region where clause
            has_cond = self.has_cond_pred(final_ctx)
            whereend = prefix_probs[:, 3] * has_cond + prefix_probs[:, 4] * (1 - has_cond)

            if outseq.dim() == 1:
                where_outseq = whereend.max(1)[1]
            else:
                where_outseq = outseq[:, 5:]

            if self.outlin.automasker is not None:
                self.outlin.automasker.update(whereend.max(1)[1])

            where_decoding = self.decoder(where_outseq, ctx=ctx, ctx_mask=inpmask, ctx_inp=inpseq,
                                    maxtime=osm.matrix.shape[1]-6)
            # endregion

            outprobs = torch.cat([prefix_probs[:, 1:3],
                                  torch.stack([select_arg_one_probs, select_arg_two_probs], 1),
                                  whereend.unsqueeze(1),
                                  where_decoding], 1)
            # endregion
            return outprobs

    class EncDecSelect(EncDec):
        """ Predicts select clause separately. Where clause the same"""
        # TODO

        def __init__(self, _inpemb, _outemb, _outlin, _encoder, dec):
            super(EncDecSelect, self).__init__(_inpemb, _outemb, _outlin, _encoder, dec)
            self.slot_ptr_addr_lin = torch.nn.Linear(innerdim, 2, bias=False)
            self.slot_ptr_sm = torch.nn.Softmax(1)
            self.has_cond_pred = torch.nn.Sequential(torch.nn.Linear(innerdim, innerdim // 2),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Linear(innerdim // 2, 1),
                                                     torch.nn.Sigmoid(), )

        def forward(self, inpseq, outseq, inpseqmaps, colnames, coltypes):
            # encoding
            self.inpemb.prepare(inpseqmaps)
            _inpembs, _inpmask = self.inpemb(inpseq)
            _inpenc = self.encoder(_inpembs, mask=_inpmask)
            inpmask = _inpmask[:, :_inpenc.size(1)]
            final_ctx = self.encoder.y_n[-1]
            final_ctx = torch.cat([final_ctx[:, 0], final_ctx[:, 1]], 1)
            inpenc = q.intercat(_inpenc.chunk(2, -1), -1)  # TODO: do we need intercat?
            ctx = inpenc  # old normalpointer mode

            # decoding
            self.outemb.prepare(inpseqmaps, colnames)
            self.outlin.prepare(inpseqmaps, colnames)

            if self.outlin.automasker is not None:
                self.outlin.automasker.update_inpseq(inpseq)
                self.outlin.automasker.update_coltypes(coltypes)

            # region predicting separately
            # region slot ptr
            slot_ptr_scores = self.slot_ptr_addr_lin(ctx)
            slot_ptr_scores = slot_ptr_scores + torch.log(inpmask.unsqueeze(2).float())
            slot_ptr_addrs = self.slot_ptr_sm(slot_ptr_scores)
            slot_ptr_one = (ctx * slot_ptr_addrs[:, :, 0:1]).sum(1)
            slot_ptr_two = (ctx * slot_ptr_addrs[:, :, 1:2]).sum(1)
            slot_ptr_one = torch.cat(
                [self.slot_ptr_addr_lin.weight[0, :].unsqueeze(0).repeat(inpseq.size(0), 1),
                 slot_ptr_one], 1)
            slot_ptr_two = torch.cat(
                [self.slot_ptr_addr_lin.weight[1, :].unsqueeze(0).repeat(inpseq.size(0), 1),
                 slot_ptr_two], 1)
            # endregion

            # region prefix probs
            prefix = ["<START>", "<QUERY>", "<SELECT>", "<WHERE>", "<END>"]
            prefix = [osm.D[p] for p in prefix]
            prefix = torch.tensor(prefix).to(inpseq.device)
            prefix_probs = torch.zeros(inpseq.size(0), prefix.size(0), max(osm.D.values()) + 1).to(
                inpseq.device)
            prefix_probs.scatter_(2, prefix.unsqueeze(0).repeat(inpseq.size(0), 1).unsqueeze(2), 1.)
            ones = torch.ones(inpseq.size(0)).to(inpseq.device)
            # endregion

            # region select clause probs
            if self.outlin.automasker is not None:
                self.outlin.automasker.update(ones.long() * osm.D["<SELECT>"])

            select_arg_one_probs = self.outlin(slot_ptr_one, None, None)
            if self.outlin.automasker is not None:
                self.outlin.automasker.update(select_arg_one_probs.max(1)[1])
            select_arg_two_probs = self.outlin(slot_ptr_two, None, None)
            # endregion

            # region where clause
            has_cond = self.has_cond_pred(final_ctx)
            whereend = prefix_probs[:, 3] * has_cond + prefix_probs[:, 4] * (1 - has_cond)

            if outseq.dim() == 1:
                where_outseq = whereend.max(1)[1]
            else:
                where_outseq = outseq[:, 5:]

            if self.outlin.automasker is not None:
                self.outlin.automasker.update(whereend.max(1)[1])

            where_decoding = self.decoder(where_outseq, ctx=ctx, ctx_mask=inpmask, ctx_inp=inpseq,
                                          maxtime=osm.matrix.shape[1] - 6)
            # endregion

            outprobs = torch.cat([prefix_probs[:, 1:3],
                                  torch.stack([select_arg_one_probs, select_arg_two_probs], 1),
                                  whereend.unsqueeze(1),
                                  where_decoding], 1)
            # endregion
            return outprobs
    # endregion

    # region decoders and model, for train and test
    def create_train_and_test_models():
        _inpemb, _outemb, _outlin, _encoder = create_submodules()
        layers = [q.LSTMCell(decdims[i-1], decdims[i], dropout_in=dropout, dropout_rec=rdropout)
                  for i in range(1, len(decdims))]
        _core = torch.nn.Sequential(*layers)
        if attmode == "dot":
            if coveragepenalty > 0.:
                _attention = q.DotAttentionWithCoverage()
                _attention.penalty.weight = coveragepenalty
            else:
                _attention = q.DotAttention()
        elif attmode == "fwd":
            _attention = q.FwdAttention(ctxdim=decdims[-1], qdim=decdims[-1], attdim=decdims[-1])
        else:
            raise q.SumTingWongException("unsupported attmode: {}".format(attmode))
        # _merger = torch.nn.Sequential(
        #     torch.nn.Linear(outlindim, outlindim),
        #     torch.nn.Tanh(),
        # )
        _merger = None
        decoder_cell = q.PointerGeneratorCell(emb=_outemb, core=_core, att=_attention, merge=_merger,
                                              out=_outlin)
        train_decoder = q.TFDecoder(decoder_cell)
        valid_decoder = q.FreeDecoder(decoder_cell)

        if useslotptr:
            print("USING SLOTPTR !!")
            _m = EncDecSlotPtr(_inpemb, _outemb, _outlin, _encoder, train_decoder)  # ONLY USE FOR TRAINING !!!
            _valid_m = EncDecSlotPtr(_inpemb, _outemb, _outlin, _encoder, valid_decoder)  # use for valid
            _valid_m.has_cond_pred = _m.has_cond_pred
            _valid_m.slot_ptr_addr_lin = _m.slot_ptr_addr_lin
        else:
            _m = EncDec(_inpemb, _outemb, _outlin, _encoder, train_decoder)         # ONLY USE FOR TRAINING !!!
            _valid_m = EncDec(_inpemb, _outemb, _outlin, _encoder, valid_decoder)     # use for valid
        return _m, _valid_m

    m, valid_m = create_train_and_test_models()
    # TODO: verify that valid_m doesn't get something wrong !
    # endregion

    # region training preparation
    trainloader = q.dataload(*traindata, batch_size=batsize, shuffle=True if not test else False)
    validloader = q.dataload(*devdata, batch_size=batsize, shuffle=False)
    testloader = q.dataload(*testdata, batch_size=batsize, shuffle=False)

    losses = q.lossarray(q.SeqKLLoss(ignore_index=0, label_smoothing=labelsmoothing, smooth_mix=smoothmix),
                         q.SeqAccuracy(ignore_index=0))

    row2tree = lambda x: SqlNode.parse_sql(osm.pp(x))

    validlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=row2tree))

    logger.update_settings(optimizer="adam")
    optim = torch.optim.Adam(q.paramgroups_of(m), lr=lr, weight_decay=wreg)

    def inp_bt(ismbatch, osmbatch, gwidsbatch, colnameids, coltypes):
        colnames = cnsm.matrix[colnameids.cpu().data.numpy()]
        colnames = torch.tensor(colnames).to(colnameids.device)
        return ismbatch, osmbatch[:, :-1], gwidsbatch, colnames, coltypes, osmbatch[:, 1:]

    def valid_inp_bt(ismbatch, osmbatch, gwidsbatch, colnameids, coltypes):
        colnames = cnsm.matrix[colnameids.cpu().data.numpy()]
        colnames = torch.tensor(colnames).to(colnameids.device)
        return ismbatch, osmbatch[:, 0], gwidsbatch, colnames, coltypes, osmbatch[:, 1:]

    if test:
        if False:
            # get model outs
            batch_in = enumerate(trainloader).next()[1]
            batch_in = inp_bt(*batch_in)
            modelouts = m(*batch_in[:-1])
            for i in range(modelouts.size(0)):
                ios = modelouts[i]
                print(i)
                print(gwids.pp(batch_in[2][i].cpu().detach().numpy()))
                print(osm.pp(batch_in[1][i].cpu().detach().numpy()))
                ats = sorted([osm.RD[x.item()] for x in ios.sum(0).nonzero()[:, 0]])
                print(" ".join(ats))
                prvnonzro = ios[0].nonzero()
                for j in range(1, ios.size(0)):
                    assert((prvnonzro == ios[j].nonzero()).all().cpu().item() == 1)

            q.embed()


    # saving best model
    best_saver = BestSaver(lambda: validlosses.get_agg_errors()[1],
                           valid_m, path=model_save_path, verbose=True)

    clip_grad_norm = q.ClipGradNorm(gradnorm)
    # endregion

    # region training
    trainer = q.trainer(m).on(trainloader).loss(losses).optimizer(optim).set_batch_transformer(inp_bt)\
        .device(device).hook(clip_grad_norm)
    validator = q.tester(valid_m).on(validloader).loss(validlosses).set_batch_transformer(valid_inp_bt)\
        .device(device).hook(best_saver)
    q.train(trainer, validator).log(logger).run(epochs=epochs)

    # q.train(m).train_on(trainloader, losses)\
    #     .optimizer(optim).clip_grad_norm(gradnorm).set_batch_transformer(inp_bt)\
    #     .valid_with(valid_m).valid_on(validloader, validlosses).set_valid_batch_transformer(valid_inp_bt)\
    #     .cuda(cuda).hook(logger).hook(best_saver)\
    #     .train(epochs)
    logger.update_settings(completed=True)

    # grad check inspection: only output layer inpemb and inpemb_trans have zero-norm grads,
    # because they're overridden in BFOL
    # endregion

    # region evaluation
    tt.tick("evaluating")

    valid_m.load_state_dict(torch.load(model_save_path))

    tt.msg("generating model from scratch")
    _, test_m = create_train_and_test_models()

    tt.msg("setting weights from best model: {}".format(model_save_path))
    test_m.load_state_dict(torch.load(model_save_path))

    valid_m.to(torch.device("cpu"))

    test_m_param_dic = {n: p for n, p in test_m.named_parameters()}
    valid_m_param_dic = {n: p for n, p in valid_m.named_parameters()}
    diffs = {}
    allzerodiffs = True
    for n in valid_m_param_dic:
        diffs[n] = (valid_m_param_dic[n] - test_m_param_dic[n]).float().norm()
        allzerodiffs &= diffs[n].cpu().item() == 0
    if not allzerodiffs:
        print("reloaded weights don't match")
        q.embed()

    if test:
        q.embed()

    # assert(all([(list(test_m.parameters())[i] - list(valid_m.parameters())[i].cpu()).float().norm()[0] == 0 for i in range(42)]))

    testlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=row2tree))
    finalvalidlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=row2tree))

    valid_results = q.tester(test_m).on(validloader).loss(finalvalidlosses)\
        .set_batch_transformer(valid_inp_bt).device(device).run()
    print("DEV RESULTS:")
    print(valid_results)
    logger.update_settings(valid_results=valid_results)
    if not test:
        test_results = q.tester(test_m).on(testloader).loss(testlosses)\
            .set_batch_transformer(valid_inp_bt).device(device).run()
        print("TEST RESULTS:")
        print(test_results)
        logger.update_settings(test_results=test_results)

    def test_inp_bt(ismbatch, osmbatch, gwidsbatch, colnameids, coltypes):
        colnames = cnsm.matrix[colnameids.cpu().data.numpy()]
        colnames = torch.tensor(colnames).to(colnameids.device)
        return ismbatch, osmbatch[:, 0], gwidsbatch, colnames, coltypes
    dev_sql_acc, test_sql_acc = evaluate_model(test_m, devdata, testdata, rev_osm_D, rev_gwids_D,
                                               inp_bt=test_inp_bt, batsize=batsize, device=device,
                                               savedir=logger.p, test=test)
    logger.update_settings(dev_sql_acc=dev_sql_acc, test_sql_acc=test_sql_acc)
    tt.tock("evaluated")
    # endregion


def run_seq2seq_oracle_df(lr=0.001, batsize=100, epochs=50,
                          inpembdim=50, outembdim=50, innerdim=100, numlayers=2, dim=-1, gdim=-1,
                          dropout=0.2, rdropout=0.1, edropout=0., idropout=0.2, irdropout=0.1,
                          wreg=1e-14, gradnorm=5., useglove=True, gfrac=0.0,
                          cuda=False, gpu=0, tag="none", ablatecopy=False, test=False,
                          tieembeddings=False, dorare=False,
                          oraclemode="zerocost", selectcolfirst=False,
                          userules="no", ptrgenmode="sharemax", labelsmoothing=0.,
                          useoffset=False, useskip=False, synonly=False,
                          uniformpretrain=-1): # oraclemode: "zerocost" or "sample"
    # region init
    settings = locals().copy()
    logger = q.Logger(prefix="wikisql_s2s_oracle_df_new")
    logger.save_settings(**settings)
    logger.update_settings(completed=False)
    print("LOGGER PATH: {}".format(logger.p))
    logger.update_settings(version="1")

    model_save_path = os.path.join(logger.p, "model")

    print("Seq2Seq + ORACLE (new)")
    print("PyTorch initial seed: {}".format(torch.initial_seed()))

    device = torch.device("cpu")
    if cuda:    device = torch.device("cuda", gpu)
    tt = q.ticktock("script")
    # endregion

    # region dimensions     # exactly the same as tf script
    gdim = None if gdim < 0 else gdim

    if dim > 0:
        innerdim = dim  # total dimension of encoding
        inpembdim = dim // 2  # dimension of input embedding
        outembdim = dim // 2  # dimension of output embedding
        if gdim is not None:
            inpembdim = gdim
            outembdim = gdim

    outlindim = innerdim * 2  # dimension of output layer - twice the encoding dim because cat of enc and state

    encdim = innerdim // 2  # half the dimension because bidirectional encoder doubles it back
    encdims = [inpembdim] + [encdim] * numlayers  # encoder's layers' dimensions
    decdims = [outembdim] + [innerdim] * numlayers  # decoder's layers' dimensions
    # endregion

    # region data
    ism, osm, cnsm, gwids, splits, e2cn = load_matrices()
    column_types = load_coltypes()
    gwids._matrix = gwids.matrix * (gwids.matrix != gwids.D["<RARE>"])
    devstart, teststart = splits
    eids = np.arange(0, len(ism), dtype="int64")
    if selectcolfirst:
        osm = reorder_select(osm)
    # splits
    if test:    devstart, teststart, batsize = 200, 250, 50
    datamats = [ism.matrix, osm.matrix, gwids.matrix, e2cn, column_types, eids]
    traindata = [datamats[i][:devstart] for i in [0, 1, 2, 3, 4, 5]]
    devdata = [datamats[i][devstart:teststart] for i in [0, 1, 2, 3, 4]]       # should be same as tf script
    testdata = [datamats[i][teststart:] for i in [0, 1, 2, 3, 4]]              # should be same as tf script

    rev_osm_D = {v: k for k, v in osm.D.items()}
    rev_gwids_D = {v: k for k, v in gwids.D.items()}

    gdic = q.PretrainedWordEmb(gdim).D
    rare_gwids_after_glove = get_rare_stats(traindata[0], traindata[2], gwids.D, gdic)
    print("{} doing rare".format("NOT" if not dorare else ""))
    if not dorare:
        rare_gwids_after_glove = None
    cnsm = do_rare_in_colnames(cnsm, traindata[-2], gdic, replace=dorare)

    # oracle:
    tracker = make_tracker_df(osm)
    # oracle = make_oracle_df(tracker, mode=oraclemode)
    # endregion

    # region submodules     # exactly the same as tf script
    def create_submodules():
        _inpemb, inpbaseemb = make_inp_emb(inpembdim, ism.D, gwids.D, useglove=useglove, gdim=gdim, gfrac=gfrac,
                                           rare_gwids=rare_gwids_after_glove)
        _outemb, inpbaseemb, colbaseemb, _ = make_out_emb(outembdim, osm.D, gwids.D, cnsm.D, gdim=gdim,
                                                          inpbaseemb=inpbaseemb, useglove=useglove, gfrac=gfrac,
                                                          rare_gwids=rare_gwids_after_glove, useskip=useskip)
        if not tieembeddings:
            inpbaseemb, colbaseemb = None, None

        automasker = None
        if userules != "no":
            automasker = MyAutoMasker(osm.D, osm.D, ctxD=ism.D, selectcolfirst=selectcolfirst, usesem=not synonly)
            automasker.test_only = userules == "test"
        _outlin, inpbaseemb, colbaseemb, colenc = make_out_lin(outlindim, ism.D, osm.D, gwids.D, cnsm.D,
                                                              useglove=useglove, gdim=gdim, gfrac=gfrac,
                                                              inpbaseemb=inpbaseemb, colbaseemb=colbaseemb,
                                                              colenc=None, nocopy=ablatecopy, rare_gwids=rare_gwids_after_glove,
                                                               automasker=automasker, useskip=useskip,
                                                               ptrgenmode=ptrgenmode, useoffset=useoffset)

        _encoder = q.FastestLSTMEncoder(*encdims, dropout_in=idropout, dropout_rec=irdropout, bidir=True)
        return _inpemb, _outemb, _outlin, _encoder
    # inpemb, outemb, outlin, encoder = create_submodules()
    # endregion

    # region encoder decoder definitions
    # -- changes from TF script: added maxtime on class itself, forward additionally takes eids and maxtime
    class EncDec(torch.nn.Module):
        def __init__(self, _inpemb, _outemb, _outlin, _encoder, dec):
            super(EncDec, self).__init__()
            self.inpemb, self.outemb, self.outlin, self.encoder, self.decoder \
                = _inpemb, _outemb, _outlin, _encoder, dec

        def forward(self, inpseq, outseq_starts, inpseqmaps, colnames, coltypes, eids=None):
            # encoding
            self.inpemb.prepare(inpseqmaps)
            _inpembs, _inpmask = self.inpemb(inpseq)
            _inpenc = self.encoder(_inpembs, mask=_inpmask)
            inpmask = _inpmask[:, :_inpenc.size(1)]
            inpenc = q.intercat(_inpenc.chunk(2, -1), -1)
            ctx = inpenc  # old normalpointer mode
            if useskip:
                ctxadd = torch.zeros(ctx.size(0), ctx.size(1), ctx.size(2) - _inpembs.size(2)).to(ctx.device)
                ctxadd = torch.cat([ctxadd, _inpembs[:, :ctx.size(1)]], 2)
                ctx = ctx + ctxadd

            # decoding
            self.outemb.prepare(inpseqmaps, colnames)
            self.outlin.prepare(inpseqmaps, colnames)

            if self.outlin.automasker is not None:
                self.outlin.automasker.update_inpseq(inpseq)
                self.outlin.automasker.update_coltypes(coltypes)

            decinp = (eids, outseq_starts) if eids is not None else outseq_starts
            decoding = self.decoder(decinp, ctx=ctx, ctx_mask=inpmask, ctx_inp=inpseq,
                                    maxtime=osm.matrix.shape[1]-1)

            return decoding
    # endregion

    # region decoders and model, for train and test
    def create_train_and_test_models():
        _inpemb, _outemb, _outlin, _encoder = create_submodules()
        layers = [q.LSTMCell(decdims[i - 1], decdims[i], dropout_in=dropout, dropout_rec=rdropout)
                  for i in range(1, len(decdims))]
        _core = torch.nn.Sequential(*layers)
        _attention = q.DotAttention()
        _merger = None
        decoder_cell = q.PointerGeneratorCell(emb=_outemb, core=_core, att=_attention, merge=_merger, out=_outlin)
        train_decoder = q.DynamicOracleDecoder(decoder_cell, tracker=tracker, mode=oraclemode)
        valid_decoder = q.FreeDecoder(decoder_cell)

        _m = EncDec(_inpemb, _outemb, _outlin, _encoder, train_decoder)
        _valid_m = EncDec(_inpemb, _outemb, _outlin, _encoder, valid_decoder)
        return _m, _valid_m

    m, valid_m = create_train_and_test_models()
    # TODO: verify that valid_m doesn't get something wrong !
    # endregion

    # region training preparation
    trainloader = q.dataload(*traindata, batch_size=batsize, shuffle=True if not test else False)
    validloader = q.dataload(*devdata, batch_size=batsize, shuffle=False)
    testloader = q.dataload(*testdata, batch_size=batsize, shuffle=False)

    losses = q.lossarray(q.SeqKLLoss(ignore_index=0, label_smoothing=labelsmoothing),
                         q.SeqAccuracy(ignore_index=0))

    row2tree = lambda x: SqlNode.parse_sql(osm.pp(x))

    validlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=row2tree))

    logger.update_settings(optimizer="adam")
    optim = torch.optim.Adam(q.paramgroups_of(m), lr=lr, weight_decay=wreg)

    def inp_bt(ismbatch, osmbatch, gwidsbatch, colnameids, coltypes, eids):
        colnames = cnsm.matrix[colnameids.cpu().data.numpy()]
        colnames = torch.tensor(colnames).to(colnameids.device)
        return ismbatch, osmbatch[:, 0], gwidsbatch, colnames, coltypes, eids, eids

    def valid_inp_bt(ismbatch, osmbatch, gwidsbatch, colnameids, coltypes):
        colnames = cnsm.matrix[colnameids.cpu().data.numpy()]
        colnames = torch.tensor(colnames).to(colnameids.device)
        return ismbatch, osmbatch[:, 0], gwidsbatch, colnames, coltypes, osmbatch[:, 1:]

    # old script had gold return whole seq but used valid_gold_bt to remove first element

    def out_bt(_out):
        return _out     #[:, :-1, :]

    def gold_bt(_eids):
        return torch.stack(m.decoder.goldacc, 1)

    # saving best model
    best_saver = BestSaver(lambda: validlosses.get_agg_errors()[1],
                           valid_m, path=model_save_path, verbose=True)

    clip_grad_norm = q.ClipGradNorm(gradnorm)
    # endregion

    # region uniform pretrain
    class PretrainHooker(q.AutoHooker):
        def get_hooks(self, ee):
            return {q.trainer.END_EPOCH: self.on_end_epoch}

        def on_end_epoch(self, _trainer, **kw):
            if _trainer.current_epoch >= uniformpretrain:
                if m.decoder.mode != oraclemode:
                    m.decoder.set_mode(oraclemode)
                    print("done pretraining uniformly")

    # endregion

    # region training
    trainer = q.trainer(m).on(trainloader).loss(losses).optimizer(optim)\
        .set_batch_transformer(inp_bt, out_bt, gold_bt)\
        .device(device).hook(clip_grad_norm)

    if uniformpretrain > 0:
        print("pretraining uniformly")
        m.decoder.set_mode("uniform")
        trainer.hook(PretrainHooker())

    validator = q.tester(valid_m).on(validloader).loss(validlosses).set_batch_transformer(valid_inp_bt)\
        .device(device).hook(best_saver)
    q.train(trainer, validator).log(logger).run(epochs=epochs)
    logger.update_settings(completed=True)
    # endregion

    # region evaluation     -- exactly same as tf script
    tt.tick("evaluating")

    valid_m.load_state_dict(torch.load(model_save_path))

    tt.msg("generating model from scratch")
    _, test_m = create_train_and_test_models()

    tt.msg("setting weights from best model: {}".format(model_save_path))
    test_m.load_state_dict(torch.load(model_save_path))

    valid_m.to(torch.device("cpu"))

    test_m_param_dic = {n: p for n, p in test_m.named_parameters()}
    valid_m_param_dic = {n: p for n, p in valid_m.named_parameters()}
    diffs = {}
    allzerodiffs = True
    for n in valid_m_param_dic:
        diffs[n] = (valid_m_param_dic[n] - test_m_param_dic[n]).float().norm()
        allzerodiffs &= diffs[n].cpu().item() == 0
    if not allzerodiffs:
        print("reloaded weights don't match")
        q.embed()

    if test:
        q.embed()

    testlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=row2tree))
    finalvalidlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=row2tree))

    valid_results = q.tester(test_m).on(validloader).loss(finalvalidlosses)\
        .set_batch_transformer(valid_inp_bt).device(device).run()
    print("DEV RESULTS:")
    print(valid_results)
    logger.update_settings(valid_seq_acc=valid_results[0], valid_tree_acc=valid_results[1])
    if not test:
        test_results = q.tester(test_m).on(testloader).loss(testlosses)\
            .set_batch_transformer(valid_inp_bt).device(device).run()
        print("TEST RESULTS:")
        print(test_results)
        logger.update_settings(test_seq_acc=test_results[0], test_tree_acc=test_results[1])

    def test_inp_bt(ismbatch, osmbatch, gwidsbatch, colnameids, coltypes):
        colnames = cnsm.matrix[colnameids.cpu().data.numpy()]
        colnames = torch.tensor(colnames).to(colnameids.device)
        return ismbatch, osmbatch[:, 0], gwidsbatch, colnames, coltypes
    dev_sql_acc, test_sql_acc = evaluate_model(test_m, devdata, testdata, rev_osm_D, rev_gwids_D,
                                               inp_bt=test_inp_bt, batsize=batsize, device=device,
                                               savedir=logger.p, test=test)
    tt.tock("evaluated")
    # endregion

# endregion

# region ERROR ANALYSIS
def compare_lines(xpath="", goldpath=DATA_PATH+"dev.gold.outlines"):
    with codecs.open(xpath, encoding="utf-8") as xf, codecs.open(goldpath, encoding="utf-8") as gf:
        i = 0
        for xline, gline in zip(xf.readlines(), gf.readlines()):
            if xline != gline:
                print(u"PREDICTION: {} \nGOLD:       {}\n".format(xline.strip(), gline.strip()))
                i += 1
        print("{} lines different".format(i))


def compare_trees(xpath="", goldpath=DATA_PATH+"dev.gold.outlines", selectcolfirst=False):
    with codecs.open(xpath, encoding="utf-8") as xf, codecs.open(goldpath, encoding="utf-8") as gf:
        i = 0
        c = 0
        select_acc = 0.
        select_agg_c = 0.
        select_col_c = 0.
        select_colagg_c = 0.
        select_c_norm = 0.
        where_acc = 0.
        both_wrong_c = 0.
        for xline, gline in zip(xf.readlines(), gf.readlines()):
            xtree = SqlNode.parse_sql(xline)
            gtree = SqlNode.parse_sql(gline)
            if selectcolfirst:
                selectnode = list(get_children_by_name(gtree, "<SELECT>"))
                assert(len(selectnode) == 1)
                selectnode = selectnode[0]
                selectchildren = selectnode.children[::-1]
                for j, selectchild in enumerate(selectchildren):
                    selectchild.order = j
                selectnode.children = selectchildren
            # print(xtree.pptree())
            # print(gtree.pptree())
            if not (gtree.equals(xtree)):
                select_wrong = False
                where_wrong = False
                both_wrong = False
                print(u"{} \nPREDICTION: {} \nGOLD:       {}\n".format(i, xline.strip(), gline.strip()))
                c += 1

                x_select_node, g_select_node = list(get_children_by_name(xtree, "<SELECT>")), list(get_children_by_name(gtree, "<SELECT>"))
                if len(x_select_node) != 1 or not x_select_node[0].equals(g_select_node[0]):
                    select_wrong = True

                if len(x_select_node) == 1 and select_wrong:
                    select_agg_wrong = False
                    select_col_wrong = False
                    x_select_agg = list(get_children_by_name(x_select_node[0], "AGG\d"))
                    g_select_agg = list(get_children_by_name(g_select_node[0], "AGG\d"))
                    if len(x_select_agg) != 1 or not x_select_agg[0].equals(g_select_agg[0]):
                        select_agg_wrong = True
                    x_select_col = list(get_children_by_name(x_select_node[0], "COL\d+"))
                    g_select_col = list(get_children_by_name(g_select_node[0], "COL\d+"))
                    if len(x_select_col) != 1 or not x_select_col[0].equals(g_select_col[0]):
                        select_col_wrong = True

                    if select_col_wrong:
                        select_col_c += 1
                    if select_agg_wrong:
                        select_agg_c += 1
                    if select_col_wrong and select_agg_wrong:
                        select_colagg_c += 1
                    select_c_norm += 1

                x_where_node, g_where_node = list(get_children_by_name(xtree, "<WHERE>")), list(get_children_by_name(gtree, "<WHERE>"))
                if len(x_where_node) != len(g_where_node):
                    where_wrong = True
                elif len(g_where_node) == 0:
                    if len(x_where_node) > 0:
                        where_wrong = True
                else:
                    if not x_where_node[0].equals(g_where_node[0]):
                        where_wrong = True
                if select_wrong:
                    select_acc -= 1
                if where_wrong:
                    where_acc -= 1
                if select_wrong and where_wrong:
                    both_wrong_c += 1
            # break
            i += 1
        print("{} trees different".format(c))
        print("{} ({}/{}) select acc".format(1.+select_acc/i, -select_acc, i))
        print("\t{} ({}/{}) select agg wrong".format(select_agg_c / (select_c_norm), select_agg_c, select_c_norm))
        print("\t{} ({}/{}) select col wrong".format(select_col_c / (select_c_norm), select_col_c, select_c_norm))
        print("\t{} ({}/{}) select both agg and col are wrong".format(select_colagg_c / (select_c_norm), select_colagg_c, select_c_norm))
        print("{} ({}/{}) where acc".format(1.+where_acc/i, -where_acc, i))
        print("{} ({}/{}) both select and where are wrong".format(both_wrong_c/i, both_wrong_c, i))
# endregion


if __name__ == "__main__":
    # jsonls_to_lines()
    # get_column_types()
    # test_matrices(writeout=True)
    # test_querylin2json()
    # test_sqlnode_and_sqls()
    # test_grouptracker()
    # test_save()
    # q.argprun(run_seq2seq_tf)
    # q.argprun(run_seq2seq_oracle_df)
    q.argprun(compare_trees)
