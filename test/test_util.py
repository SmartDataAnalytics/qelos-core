from unittest import TestCase
import qelos_core as q
import torch
import numpy as np


class TestMasked_mean(TestCase):
    def test_it_full_mask(self):
        x = torch.randn(2,3,4)
        mask = torch.ones_like(x)[:, :, 0].unsqueeze(-1)
        for i in range(3):
            print(i)
            y = q.masked_mean(x, mask=mask, dim=i)
            y_ref = torch.mean(x, dim=i)
            self.assertTrue(np.allclose(y.detach().numpy(), y_ref.detach().numpy()))

    def test_it_easy_mask(self):
        x = torch.randn(5,6)
        mask = torch.ones_like(x)
        mask[:, [3,4,5]] = 0
        # print(mask)
        y = q.masked_mean(x, 1, mask=mask)
        print(y)
        y_ref = torch.mean(x[:, :3], 1)
        print(y_ref)
        self.assertTrue(np.allclose(y.detach().numpy(), y_ref.detach().numpy()))

        y = q.masked_mean(x, 0, mask=mask)
        y_ref = torch.mean(x*mask, 0)
        print(y)
        print(y_ref)
        self.assertTrue(np.allclose(y.detach().numpy(), y_ref.detach().numpy()))