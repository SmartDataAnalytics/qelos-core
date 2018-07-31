import qelos_core as q
import torch
import json
import pickle
import numpy as np


DATA_PATH = "../../../datasets/convai2/"


def load_data(p=DATA_PATH):
    train_struct = json.load(open(p+"train_dialogues.struct.json"))
    valid_struct = json.load(open(p+"valid_dialogues.struct.json"))
    D = json.load(open(p+"dialogues.strings.dict"))
    mat = np.load(p+"dialogues.strings.mat.npy")
    return train_struct, valid_struct, mat, D


def load_datasets(p=DATA_PATH):
    train_struct, valid_struct, mat, D = load_data(p)
    train_dataset = ConvAI2Dataset(train_struct, mat, D)
    valid_dataset = ConvAI2Dataset(valid_struct, mat, D)
    return train_dataset, valid_dataset


class ConvAI2Dataset(torch.utils.data.Dataset):
    def __init__(self, struct=None, mat=None, D=None, **kw):
        super(ConvAI2Dataset, self).__init__(**kw)
        self.struct, self.mat, self.D = struct, mat, D
        self.numchoices = 20
        self.numpersonalines = 5
        self.numturns = 16

    def __len__(self):
        return len(self.struct)

    def __getitem__(self, i):
        """
        Returns three tensors:  * self persona for this example
                                * other persona for this example
                                * dialogue for this example
        """
        selfpersona_mat = np.zeros((self.numpersonalines, self.mat.shape[1]), dtype="int64")
        for j, sid in enumerate(self.struct[i]["self"]):
            selfpersona_mat[j, :] = self.mat[sid]
        otherpersona_mat = np.zeros((self.numpersonalines, self.mat.shape[1]), dtype="int64")
        for j, sid in enumerate(self.struct[i]["other"]):
            otherpersona_mat[j, :] = self.mat[sid]

        selflines_mat = np.zeros((self.numturns, self.mat.shape[1]), dtype="int64")
        otherlines_mat = np.zeros((self.numturns, self.mat.shape[1]), dtype="int64")

        selfchoices = []
        for j, turn in enumerate(self.struct[i]["lines"]):
            if "other" in turn:
                otherlines_mat[j//2, :] = self.mat[turn["other"]]
            elif "self" in turn:
                selflines_mat[j//2, :] = self.mat[turn["self"]]
                selfchoices.append(turn["self_choices"])

        selfchoices_mat = np.zeros((self.numturns, 20, self.mat.shape[1]), dtype="int64")
        for i, selfchoices_e in enumerate(selfchoices):
            for j, sid in enumerate(selfchoices_e):
                selfchoices_mat[i, j, :] = self.mat[sid]

        return otherpersona_mat, selfpersona_mat, otherlines_mat, selflines_mat, selfchoices_mat




def run(lr=0.001,
        batsize=10,
        test=False,
        ):
    tt = q.ticktock("script")
    tt.tick("loading data")
    train_dataset, valid_dataset = load_datasets()
    tt.tock("loaded data")
    print("{} unique words, {} training examples, {} valid examples".format(len(train_dataset.D), len(train_dataset), len(valid_dataset)))
    trainloader = q.dataload(train_dataset, shuffle=True, batch_size=batsize)
    validloader = q.dataload(valid_dataset, shuffle=True, batch_size=batsize)
    # test
    if test:
        testexample = train_dataset[10]
        trainloader_iter = iter(trainloader)
        tt.tick("getting 1000 batches")
        for i in range(1000):
            batch = next(iter(trainloader))
        tt.tock("got 1000 batches")

    print("done")




if __name__ == "__main__":
    q.argprun(run)