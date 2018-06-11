from unittest import TestCase
import qelos_core as q
import torch
import numpy as np


class Test_seq_pack_unpack(TestCase):
    def test_it(self):
        seq = torch.randn(5, 10, 4)
        lens = np.random.randint(1, 10, (5,))
        mask = np.zeros((5, 10))
        for i in range(mask.shape[0]):
            mask[i, :lens[i]] = 1
        print(mask)
        mask = torch.tensor(mask, dtype=torch.uint8)
        o, us = q.seq_pack(seq, mask)
        # print(seq[:, :, 0])
        # print(o[:, :, 0])
        # print(o)
        print(us)
        recons, remask = q.seq_unpack(o, us)
        print(recons)
        seq = seq.cpu().detach().numpy()
        seq *= mask.cpu().detach().numpy()[:, :, np.newaxis]
        recons = recons.cpu().detach().numpy()
        self.assertTrue(np.allclose(seq[:, :recons.shape[1]], recons))
