import qelos_core as q
import torch
import numpy as np


# Hierarchical Encoder Decoder

class BasicHierarchicalEncoderDecoder(torch.nn.Module):
    """ Models whole dialog, teacher forced generation"""
    def __init__(self, emb, outlin, embdim, encdims, diadims, decdims, **kw):
        super(BasicHierarchicalEncoderDecoder, self).__init__()
        self.emb = emb
        self.utterance_enc = q.FastestLSTMEncoder(embdim, *encdims)
        self.dialog_enc = q.FastestLSTMEncoder(encdims[-1], *diadims)
        self.utterance_dec = q.FastestLSTMEncoder(embdim + diadims[-1], *decdims)
        self.outlin = outlin
        self.sm = torch.nn.Softmax(-1)

    def forward(self, x, y):
        """
        :param x:   (batsize, nr_turns, seqlen)^vocsize
        """
        # seq packing might fail due to completely masked turns  --> TODO: fix in rnn.py
        # encode all utterances independently
        all_utt_emb, all_utt_mask = self.emb(x)

        #grad_check_param = torch.nn.Parameter(torch.zeros_like(all_utt_emb))
        #all_utt_emb = all_utt_emb + grad_check_param

        all_utt_emb = all_utt_emb.view(-1, x.size(-1), all_utt_emb.size(-1))
        _all_utt_mask = all_utt_mask.view(-1, x.size(-1))
        all_utt_outs, all_out_states = self.utterance_enc(all_utt_emb, mask=_all_utt_mask, ret_states=True)
        all_out_states = all_out_states[-1][0][:, 0, :]
        all_out_states = all_out_states.view(x.size(0), x.size(1), -1)
        all_utt_outs = all_utt_outs.view(x.size(0), x.size(1), x.size(2), -1)
        # shaping back into x's shape seems to work (checked couple gradients)

        # all_out_states has shape (batsize, nr_turns, encdim_utt)
        # now encode with dialog_enc
        dia_mask = (all_utt_mask.sum(2) > 0).float()        # 2D
        dia_states = self.dialog_enc(all_out_states, mask=dia_mask)     # (batsize, nr_turns, encdim_dia)

        # out utterances
        all_oututt_emb, all_oututt_mask = self.emb(y)       # (batsize, nr_turns, seqlen, embdim)
        _dia_states = dia_states.unsqueeze(2).expand(-1, -1, y.size(2), -1)
        all_out_inps = torch.cat([all_oututt_emb, _dia_states], 3)
        _all_out_inps = all_out_inps.view(-1, y.size(-1), all_out_inps.size(-1))
        _all_out_inps_mask = all_oututt_mask.view(-1, y.size(-1))
        dec_outs = self.utterance_dec(_all_out_inps, mask=_all_out_inps_mask)     #(batsize, nr_turns*seqlen, outvocsize)

        dec_out_scores = self.outlin(dec_outs)

        dec_out_probs = self.sm(dec_out_scores)

        _dec_out_probs = dec_out_probs.view(y.size(0), y.size(1), y.size(2), dec_out_probs.size(-1))

        # TODO: check grads from decoder onto enc utt embs

        # to get every second:
        ds = _dec_out_probs.size()
        _sel_from0_by2 = _dec_out_probs.view(ds[0], ds[1]//2, 2, ds[2], ds[3])[:, :, 0, :, :]
        _sel_from1_by2 = _dec_out_probs.view(ds[0], ds[1] // 2, 2, ds[2], ds[3])[:, :, 1, :, :]
        return _dec_out_probs


def run(lr=0.001):
    x = torch.randint(1, 100, (5, 8, 6), dtype=torch.int64)
    y = x[:, 1:, :-1]
    y = torch.cat([torch.ones(y.size(0), y.size(1), 1, dtype=y.dtype), y], 2)
    y = torch.cat([y, torch.randint(1, 100, (y.size(0), 1, y.size(2))).long()], 1)
    D = dict(zip(["<MASK>"] + [str(i) for i in range(1, 100)], range(100)))
    m = BasicHierarchicalEncoderDecoder(q.WordEmb(10, worddic=D),
                                        q.WordLinout(25, worddic=D),
                                        10,
                                        (20,),
                                        (30,),
                                        (25,))
    pred = m(x, y)


if __name__ == "__main__":
    q.argprun(run)