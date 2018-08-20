import torch
import qelos_core as q
from unittest import TestCase
import numpy as np


class TestPointerGeneratorOut(TestCase):
    def test_normal(self):
        dim = 10
        batsize = 3
        seqlen = 6

        outwords = "a b c d e 1 2 3 4".split()
        outdic = dict(zip(outwords, range(len(outwords))))

        inpdic = "1 2 3 4".split()
        inpdic = dict(zip(inpdic, range(len(inpdic))))

        gendic = "a b c d e".split()
        gendic = dict(zip(gendic, range(len(gendic))))

        gen_out = torch.nn.Sequential(
            torch.nn.Linear(dim, len(gendic)),
            torch.nn.Sigmoid())

        gen_prob_comp = torch.nn.Sequential(
            torch.nn.Linear(dim, 1),
            torch.nn.Sigmoid())

        pgo = q.PointerGeneratorOutSeparate(outdic, gen_prob_comp, gen_out, inpdic=inpdic, gen_outD=gendic)

        x = torch.randn(batsize, dim)
        scores = torch.randn(batsize, seqlen)
        ctx_inp = torch.randint(0, len(inpdic), (batsize, seqlen)).long()

        print(pgo.gen_to_out)
        print(pgo.inp_to_out)
        self.assertTrue((pgo.gen_to_out == torch.tensor([0,1,2,3,4])).all().item() == 1)
        self.assertTrue((pgo.inp_to_out == torch.tensor([5,6,7,8])).all().item() == 1)

        out = pgo(x, scores, ctx_inp)
        print(out.size())
        self.assertTrue(out.size() == (batsize, len(outdic)))
        print(out.sum(1))
        print(torch.ones_like(out[:, 0]))

        self.assertTrue(np.allclose(out.sum(1).detach().numpy(),
                        torch.ones_like(out[:, 0]).detach().numpy()))

    def test_masked(self):
        dim = 10
        batsize = 3
        seqlen = 6

        outwords = "a b c d e 1 2 3 4".split()
        outdic = dict(zip(outwords, range(len(outwords))))

        inpdic = "1 2 3 4".split()
        inpdic = dict(zip(inpdic, range(len(inpdic))))

        gendic = "a b c d e".split()
        gendic = dict(zip(gendic, range(len(gendic))))

        gen_prob_comp = torch.nn.Sequential(
            torch.nn.Linear(dim, 1),
            torch.nn.Sigmoid(),)

        gen_out = torch.nn.Linear(dim, len(gendic))

        pgo = q.PointerGeneratorOutSeparate(outdic, gen_prob_comp, gen_out, inpdic=inpdic, gen_outD=gendic)

        x = torch.randn(batsize, dim)
        scores = torch.randn(batsize, seqlen)
        ctx_inp = torch.randint(0, len(inpdic), (batsize, seqlen)).long()
        mask = torch.tensor([[1,1,0,0,1,0,1,0,1],
                             [1,1,1,1,1,0,0,0,0],
                             [0,0,0,0,0,1,1,1,1]])

        # print(pgo.gen_to_out)
        # print(pgo.inp_to_out)
        self.assertTrue((pgo.gen_to_out == torch.tensor([0,1,2,3,4])).all().item() == 1)
        self.assertTrue((pgo.inp_to_out == torch.tensor([5,6,7,8])).all().item() == 1)

        out = pgo(x, scores, ctx_inp, mask=mask)

        print(ctx_inp)
        print(out)

        print(out.size())
        self.assertTrue(out.size() == (batsize, len(outdic)))
        # print(out.sum(1))
        # print(torch.ones_like(out[:, 0]))

        self.assertTrue(np.allclose(out.sum(1).detach().numpy(),
                        torch.ones_like(out[:, 0]).detach().numpy()))


class TestAutoMasker(TestCase):
    def test_it(self):
        words = "a b c d e f g".split()
        D = dict(zip(words, range(len(words))))

        rules = {"a": ["a", "b"],
                 "b": ["b", "c"],
                 "c": ["c", "d"],
                 "d": ["d", "e"],
                 "e": ["e", "f"],
                 "f": ["f", "g"],
                 "g": ["g", "a"]}

        class MyAutoMasker(q.AutoMasker):
            def get_out_tokens_for_history(self, i, hist):
                prev = hist[-1]
                ret = rules[prev]
                return ret

        mam = MyAutoMasker(D, D)

        mam.update(torch.arange(0, max(D.values()) + 1))
        print(mam.get_out_mask())
        ref = np.eye(7)
        ref = ref + np.roll(ref, 1, axis=1)

        self.assertTrue(np.allclose(ref, mam.get_out_mask().detach().numpy()))