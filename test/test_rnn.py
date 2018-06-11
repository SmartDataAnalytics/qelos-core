import torch
import qelos_core as q
from unittest import TestCase
import numpy as np


# region TEST FASTEST LSTM
class TestFastestLSTM(TestCase):
    def setUp(self):
        batsize = 5
        seqlen = 4
        lstm = q.FastestLSTMEncoder(20, 26, 30)

        x = torch.nn.Parameter(torch.randn(batsize, seqlen, 20))

        y = lstm(x)
        self.batsize = batsize
        self.seqlen = seqlen
        self.x = x
        self.y = y
        self.lstm = lstm

    def test_shapes(self):
        self.assertEqual((self.batsize, self.seqlen, 30), self.y.detach().numpy().shape)

    def test_grad(self):
        l = self.y[2, :, :].sum()
        l.backward()

        xgrad = self.x.grad.detach().numpy()

        # no gradient to examples that weren't used for loss
        self.assertTrue(np.allclose(xgrad[:2], np.zeros_like(xgrad[:2])))
        self.assertTrue(np.allclose(xgrad[3:], np.zeros_like(xgrad[3:])))

        # gradient on the example that was used for loss
        self.assertTrue(np.linalg.norm(xgrad) > 0)

        print(xgrad[:, 0, :7])

    def test_final_states(self):
        y_T = self.lstm.y_n[-1][:, 0]
        self.assertTrue(np.allclose(y_T.detach().numpy(), self.y[:, -1].detach().numpy()))


class TestFastestLSTMInitStates(TestCase):
    def test_init_states(self):
        batsize = 5
        seqlen = 4
        lstm = q.FastestLSTMEncoder(20, 26, 30)
        lstm.train(False)
        x = torch.nn.Parameter(torch.randn(batsize, seqlen*2, 20))
        y_whole = lstm(x)

        y_first, states = lstm(x[:, :seqlen], ret_states=True)
        states = zip(*states)
        y_second = lstm(x[:, seqlen:], y_0s=states[0], c_0s=states[1])

        y_part = torch.cat([y_first, y_second], 1)

        self.assertTrue(np.allclose(y_whole.detach().numpy(), y_part.detach().numpy()))


class TestFastestLSTMBidir(TestCase):
    def setUp(self):
        batsize = 5
        seqlen = 4
        lstm = q.FastestLSTMEncoder(20, 26, 30, bidir=True)

        x = torch.nn.Parameter(torch.randn(batsize, seqlen, 20))

        y = lstm(x)
        self.batsize, self.seqlen = batsize, seqlen
        self.x, self.y = x, y
        self.lstm = lstm

    def test_shapes(self):
        self.assertEqual((self.batsize, self.seqlen, 30*2), self.y.detach().numpy().shape)

    def test_grad(self):
        l = self.y[2, :, :].sum()
        l.backward()

        xgrad = self.x.grad.detach().numpy()

        # no gradient to examples that weren't used for loss
        self.assertTrue(np.allclose(xgrad[:2], np.zeros_like(xgrad[:2])))
        self.assertTrue(np.allclose(xgrad[3:], np.zeros_like(xgrad[3:])))

        # gradient on the example that was used for loss
        self.assertTrue(np.linalg.norm(xgrad) > 0)

        print(xgrad[:, 0, :7])

    def test_final_states(self):
        y_T = self.lstm.y_n[-1]
        self.assertTrue(np.allclose(y_T[:, 0].detach().numpy(), self.y[:, -1, :30].detach().numpy()))
        self.assertTrue(np.allclose(y_T[:, 1].detach().numpy(), self.y[:, 0, 30:].detach().numpy()))
        print(y_T.size())


class TestFastestLSTMBidirMasked(TestCase):
    def setUp(self):
        batsize = 5
        seqlen = 8
        lstm = q.FastestLSTMEncoder(20, 26, 30, bidir=True)

        x = torch.nn.Parameter(torch.randn(batsize, seqlen, 20))
        mask = np.zeros((batsize, seqlen)).astype("int64")
        mask[0, :3] = 1
        mask[1, :] = 1
        mask[2, :5] = 1
        mask[3, :1] = 1
        mask[4, :4] = 1
        mask = torch.tensor(mask)

        y = lstm(x, mask=mask)
        self.batsize, self.seqlen = batsize, seqlen
        self.x, self.y = x, y
        self.mask = mask
        self.lstm = lstm

    def test_shapes(self):
        self.assertEqual((self.batsize, self.seqlen, 30*2), self.y.detach().numpy().shape)

    def test_grad(self):
        l = self.y[2, :, :].sum()
        l.backward()

        xgrad = self.x.grad.detach().numpy()

        # no gradient to examples that weren't used for loss
        self.assertTrue(np.allclose(xgrad[:2], np.zeros_like(xgrad[:2])))
        self.assertTrue(np.allclose(xgrad[3:], np.zeros_like(xgrad[3:])))

        # gradient on the example that was used for loss
        self.assertTrue(np.linalg.norm(xgrad) > 0)

        print(xgrad[2, :, :3])

    def test_grad_masked(self):
        l = self.y.sum()
        l.backward()

        xgrad = self.x.grad.detach().numpy()

        mask = self.mask.detach().numpy()[:, :, np.newaxis]

        self.assertTrue(np.linalg.norm(xgrad * mask) > 0)
        self.assertTrue(np.allclose(xgrad * (1 - mask), np.zeros_like(xgrad)))

    def test_output_mask(self):
        mask = self.mask.detach().numpy()[:, :, np.newaxis]

        self.assertTrue(np.linalg.norm(self.y.detach().numpy() * mask) > 0)
        self.assertTrue(np.allclose(self.y.detach().numpy() * (1 - mask), np.zeros_like(self.y.detach().numpy())))

    def test_final_states(self):
        y_T = self.lstm.y_n[-1]
        self.assertTrue(np.allclose(y_T[0, 0].detach().numpy(), self.y[0, 2, :30].detach().numpy()))
        self.assertTrue(np.allclose(y_T[1, 0].detach().numpy(), self.y[1, -1, :30].detach().numpy()))
        self.assertTrue(np.allclose(y_T[2, 0].detach().numpy(), self.y[2, 4, :30].detach().numpy()))
        self.assertTrue(np.allclose(y_T[3, 0].detach().numpy(), self.y[3, 0, :30].detach().numpy()))
        self.assertTrue(np.allclose(y_T[4, 0].detach().numpy(), self.y[4, 3, :30].detach().numpy()))
        self.assertTrue(np.allclose(y_T[:, 1].detach().numpy(), self.y[:, 0, 30:].detach().numpy()))
        print(y_T.size())


class TestFastestLSTMBidirMaskedWithDropout(TestCase):
    def test_dropout_in(self):
        batsize = 5
        seqlen = 8
        lstm = q.FastestLSTMEncoder(20, 30, bidir=False, dropout_in=0.3, dropout_rec=0.)

        x = torch.nn.Parameter(torch.randn(batsize, seqlen, 20))
        mask = np.zeros((batsize, seqlen)).astype("int64")
        mask[0, :3] = 1
        mask[1, :] = 1
        mask[2, :5] = 1
        mask[3, :1] = 1
        mask[4, :4] = 1
        mask = torch.tensor(mask)

        assert(lstm.training)

        y = lstm(x, mask=mask)
        self.batsize, self.seqlen = batsize, seqlen
        self.x, self.y = x, y
        self.mask = mask
        self.lstm = lstm

        y_t0_r0 = self.lstm(self.x, mask=self.mask)[:, 0, :30]
        y_t0_r1 = self.lstm(self.x, mask=self.mask)[:, 0, :30]
        y_t0_r2 = self.lstm(self.x, mask=self.mask)[:, 0, :30]

        self.assertTrue(not np.allclose(y_t0_r0.detach().numpy(), y_t0_r1.detach().numpy()))
        self.assertTrue(not np.allclose(y_t0_r1.detach().numpy(), y_t0_r2.detach().numpy()))
        self.assertTrue(not np.allclose(y_t0_r0.detach().numpy(), y_t0_r2.detach().numpy()))

# endregion