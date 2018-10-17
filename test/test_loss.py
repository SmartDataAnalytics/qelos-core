from unittest import TestCase

import numpy as np
import torch
from qelos_core.loss import SeqDistillLoss, nan2zero
import random


class TestSeqDistillLoss(TestCase):
    def test_nan2zero(self):
        x = torch.randn(5)
        x[0] = np.nan
        x.requires_grad = True
        y = nan2zero(x)
        l = y.sum()
        l.backward()
        print("backwarded")
        print(x.grad)

        try:
            x = torch.randn(5)
            x[0] = np.nan
            x.requires_grad = True
            x[x != x] = 0
            l = x.sum()
            l.backward()
        except Exception as e:
            print("didn't backward")
        #
        # x = torch.randn(5)
        # x[0] = np.nan
        # x.requires_grad = True
        # y = torch.zeros_like(x)
        # y[x == x] = x
        # l = y.sum()
        # l.backward()

    def test_it(self):
        m = SeqDistillLoss(temperature=2., soft_gold_mode="logits", ignore_index=0)
        probs = torch.randn(2, 3, 4)
        softgold = torch.randn(2, 3, 4)
        hardgold = torch.randint(1, 4, (2, 3)).to(torch.int64)
        hardgold[0, 1] = random.choice((1, 2))
        hardgold[:, -1] = 0
        softgold[0, 1, 1:3] = -np.infty
        probs[0, -1, 0] = -np.infty
        probs[0, 1, [2, 3]] = -np.infty
        l = m(probs, (softgold, hardgold))
        print(l)
        if hardgold[0, 1].item() in (2, 3):
            print("l is infty")
            self.assertTrue(l.item() == np.infty)
        else:
            print("l is not infty")
            self.assertFalse(l.item() == np.infty)
