import qelos_core as q
import torch
import re
import numpy as np
import json
from collections import OrderedDict


DATA_PATH = "../../../datasets/convai2/"


def run(p1=DATA_PATH+"valid_dialogues.json",
        p2=DATA_PATH+"valid_dialogues.json",        # change the file paths to use train and valid (so the ids are shared)
        maxwords=int(1e9), rarefreq=0):
    """
    Saves in DATA_PATH, see code for exact paths
    :param p1:          path to train json
    :param p2:          path to valid json
    :param maxwords:    maximum number of words in vocab
    :param rarefreq:    word frequency for rare words
    :return:
    """
    sm = q.StringMatrix(topnwords=maxwords, freqcutoff=rarefreq)
    sm.tokenize = lambda x: x.split()
    out_struct1, sm, us = load_datafile(p1, sm)
    sm.unseen_mode = True
    out_struct2, sm, us2 = load_datafile(p2, sm, uniquestrings=us)
    sm.finalize()
    ## !!! dictionary is in sm.D, numpy array is in sm.matrix
    assert(us == us2)
    print("done: {} unique strings \n\n".format(len(us)))
    json.dump(out_struct1, open(DATA_PATH + "train_dialogues.struct.json", "w"))
    json.dump(out_struct2, open(DATA_PATH + "valid_dialogues.struct.json", "w"))
    json.dump(sm.D, open(DATA_PATH+"dialogues.strings.dict", "w"))
    np.save(DATA_PATH+"dialogues.strings.mat", sm.matrix)
    print("saved")
    return out_struct1, out_struct2, sm


def load_datafile(p="../../../datasets/convai2/valid_dialogues.json", sm=None, uniquestrings=None):
    d = json.load(open(p))

    uniquestrings = OrderedDict() if uniquestrings is None else uniquestrings

    def add_string(l):
        if l not in uniquestrings:
            id = len(uniquestrings)
            uniquestrings[l] = id
            sm.add(l)
        else:
            id = uniquestrings[l]
        return id

    def reccer(s, mapper):
        if isinstance(s, dict):
            out = {}
            for k, v in s.items():
                out[k] = mapper(v, reccer)
            return out
        elif isinstance(s, list):
            out = []
            for e in s:
                out.append(mapper(e, reccer))
            return out
        else:
            raise NotImplemented("qsdf")

    def get_map(v, _reccer):
        if q.isstring(v):
            ret = add_string(v)
        else:
            ret = _reccer(v, get_map)
        return ret

    def get_unmap(v, _reccer):
        if isinstance(v, int):
            ret = sm[v]
        else:
            ret = _reccer(v, get_unmap)
        return ret

    out_struct = reccer(d, get_map)
    return out_struct, sm, uniquestrings


if __name__ == "__main__":
    q.argprun(run)