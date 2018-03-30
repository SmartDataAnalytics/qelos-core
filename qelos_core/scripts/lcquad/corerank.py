import qelos_core as q
import torch
import numpy as np


OPT_LR = 0.001


# TODO: port metrics


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
    q.argprun(run)