import qelos_core as q
import torch
import re
import numpy as np
import json
from collections import OrderedDict


def run(p1="../../../datasets/convai2/valid_dialogues.json",
        p2="../../../datasets/convai2/valid_dialogues.json",
        maxwords=800, rarefreq=0):
    sm = q.StringMatrix(topnwords=maxwords, freqcutoff=rarefreq)
    sm.tokenize = lambda x: x.split()
    out_struct, sm = load_datafile(p1, sm)
    sm.unseen_mode = True
    out_struct, sm = load_datafile(p2, sm)
    sm.finalize()
    print("done\n\n")
    return out_struct, sm


def load_datafile(p="../../../datasets/convai2/valid_dialogues.json", sm=None):
    d = json.load(open(p))

    uniquestrings = OrderedDict()

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
    return out_struct, sm


if __name__ == "__main__":
    q.argprun(run)