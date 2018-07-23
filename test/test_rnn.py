import torch
import qelos_core as q
from unittest import TestCase
import numpy as np


# region TEST CELLS
class TestGRUCell(TestCase):
    def test_gru_shapes(self):
        batsize = 5
        gru = q.GRUCell(9, 10)
        x_t = torch.randn(batsize, 9)
        h_tm1 = torch.randn(1, 10)
        gru.h_0 = q.val(h_tm1).v
        y_t = gru(x_t)
        self.assertEqual((batsize, 10), y_t.detach().numpy().shape)

    def test_dropout_rec(self):
        batsize = 5
        gru = q.GRUCell(9, 10, dropout_rec=0.5)
        x_t = torch.randn(batsize, 9)
        h_tm1 = torch.randn(1, 10)
        gru.h_0 = q.val(h_tm1).v
        y_t = gru(x_t)
        self.assertEqual((5, 10), y_t.detach().numpy().shape)

        self.assertEqual(gru.training, True)
        gru.train(False)
        self.assertEqual(gru.training, False)

        gru.rec_reset()
        pred1 = gru(x_t)
        gru.rec_reset()
        pred2 = gru(x_t)

        self.assertTrue(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))

        gru.train(True)
        self.assertEqual(gru.training, True)

        gru.rec_reset()
        pred1 = gru(x_t)
        gru.rec_reset()
        pred2 = gru(x_t)

        self.assertFalse(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))

    def test_zoneout(self):
        batsize = 5
        gru = q.GRUCell(9, 10, zoneout=0.5)
        x_t = torch.randn(batsize, 9)
        h_tm1 = torch.randn(1, 10)
        gru.h_0 = q.val(h_tm1).v
        y_t = gru(x_t)
        self.assertEqual((5, 10), y_t.detach().numpy().shape)

        self.assertEqual(gru.training, True)
        gru.train(False)
        self.assertEqual(gru.training, False)

        gru.rec_reset()
        pred1 = gru(x_t)
        gru.rec_reset()
        pred2 = gru(x_t)

        self.assertTrue(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))

        gru.train(True)
        self.assertEqual(gru.training, True)

        gru.rec_reset()
        pred1 = gru(x_t)
        gru.rec_reset()
        pred2 = gru(x_t)

        self.assertFalse(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))

    def test_mask_t(self):
        batsize = 5
        gru = q.GRUCell(9, 10)
        x_t = torch.randn(batsize, 9)
        mask_t = torch.tensor([1, 1, 0, 1, 0])
        h_tm1 = torch.randn(1, 10)
        gru.h_0 = q.val(h_tm1).v
        y_t = gru(x_t, mask_t=mask_t)
        self.assertEqual((batsize, 10), y_t.detach().numpy().shape)

        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), y_t[2].detach().numpy()))
        self.assertFalse(np.allclose(h_tm1[0].detach().numpy(), y_t[1].detach().numpy()))
        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), y_t[4].detach().numpy()))

        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), gru.h_tm1[2].detach().numpy()))
        self.assertFalse(np.allclose(h_tm1[0].detach().numpy(), gru.h_tm1[1].detach().numpy()))
        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), gru.h_tm1[4].detach().numpy()))


class TestLSTM(TestCase):
    def test_lstm_shapes(self):
        batsize = 5
        lstm = q.LSTMCell(9, 10)
        x_t = torch.randn(batsize, 9)
        c_tm1 = torch.randn(1, 10)
        y_tm1 = torch.randn(1, 10)
        lstm.c_0 = q.val(c_tm1).v
        lstm.y_0 = q.val(y_tm1).v

        y_t = lstm(x_t)
        self.assertEqual((5, 10), y_t.detach().numpy().shape)

        self.assertTrue(np.allclose(lstm.y_tm1.detach().numpy(), y_t.detach().numpy()))

        q.rec_reset(lstm)

    def test_dropout_rec(self):
        batsize = 5
        lstm = q.LSTMCell(9, 10, dropout_rec=0.5)
        x_t = torch.randn(batsize, 9)
        c_tm1 = torch.randn(1, 10)
        y_tm1 = torch.randn(1, 10)
        lstm.c_0 = q.val(c_tm1).v
        lstm.y_0 = q.val(y_tm1).v
        y_t = lstm(x_t)
        self.assertEqual((5, 10), y_t.detach().numpy().shape)

        self.assertEqual(lstm.training, True)
        lstm.train(False)
        self.assertEqual(lstm.training, False)

        lstm.rec_reset()
        pred1 = lstm(x_t)
        lstm.rec_reset()
        pred2 = lstm(x_t)

        self.assertTrue(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))

        lstm.train(True)
        self.assertEqual(lstm.training, True)

        lstm.rec_reset()
        pred1 = lstm(x_t)
        lstm.rec_reset()
        pred2 = lstm(x_t)

        self.assertFalse(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))

    def test_zoneout(self):
        batsize = 5
        lstm = q.LSTMCell(9, 10, zoneout=0.5)
        x_t = torch.randn(batsize, 9)
        c_tm1 = torch.randn(1, 10)
        y_tm1 = torch.randn(1, 10)
        lstm.c_0 = q.val(c_tm1).v
        lstm.y_0 = q.val(y_tm1).v
        y_t = lstm(x_t)
        self.assertEqual((5, 10), y_t.detach().numpy().shape)

        self.assertEqual(lstm.training, True)
        lstm.train(False)
        self.assertEqual(lstm.training, False)

        lstm.rec_reset()
        pred1 = lstm(x_t)
        lstm.rec_reset()
        pred2 = lstm(x_t)

        self.assertTrue(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))

        lstm.train(True)
        self.assertEqual(lstm.training, True)

        lstm.rec_reset()
        pred1 = lstm(x_t)
        lstm.rec_reset()
        pred2 = lstm(x_t)

        self.assertFalse(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))

    def test_mask_t(self):
        batsize = 5
        lstm = q.LSTMCell(9, 10)
        x_t = torch.randn(batsize, 9)
        mask_t = torch.tensor([1, 1, 0, 1, 0])
        c_tm1 = torch.randn(1, 10)
        h_tm1 = torch.randn(1, 10)
        lstm.c_0 = q.val(c_tm1).v
        lstm.y_0 = q.val(h_tm1).v
        y_t = lstm(x_t, mask_t=mask_t)
        self.assertEqual((batsize, 10), y_t.detach().numpy().shape)

        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), y_t[2].detach().numpy()))
        self.assertFalse(np.allclose(h_tm1[0].detach().numpy(), y_t[1].detach().numpy()))
        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), y_t[4].detach().numpy()))

        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), lstm.y_tm1[2].detach().numpy()))
        self.assertFalse(np.allclose(h_tm1[0].detach().numpy(), lstm.y_tm1[1].detach().numpy()))
        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), lstm.y_tm1[4].detach().numpy()))
# endregion


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


class TestFastestLSTMwithMask(TestCase):
    def test_it(self):
        batsize = 3
        seqlen = 4
        lstm = q.FastestLSTMEncoder(8, 9, 10)

        x = torch.nn.Parameter(torch.randn(batsize, seqlen, 8))
        x_mask = torch.tensor([[1,1,1,0],[1,0,0,0],[1,1,0,0]], dtype=torch.int64)

        y, states = lstm(x, mask=x_mask, ret_states=True)

        l = states[0][0][1].sum()
        l.backward()

        self.assertTrue(x.grad[0].norm() == 0)
        self.assertTrue(x.grad[1].norm() > 0)
        self.assertTrue(x.grad[2].norm() == 0)

        print("done")


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


# region attention
class TestAttention(TestCase):
    def test_dot_attention(self):
        x = torch.randn(5, 10, 7)
        y = torch.randn(5, 7)

        a = q.DotAttention()

        alphas, summaries = a(y, x)
        print(alphas.size())
        # test shapes
        self.assertEqual(alphas.size(), (5, 10))
        self.assertTrue(np.allclose(alphas.sum(1).detach().numpy(), np.ones_like(alphas.sum(1).detach().numpy())))
        self.assertEqual(summaries.size(), (5, 7))

    def test_fwd_attention(self):
        x = torch.randn(5, 10, 7)
        y = torch.randn(5, 7)

        a = q.FwdAttention(7, 7, 7)

        alphas, summaries = a(y, x)
        print(alphas.size())
        # test shapes
        self.assertEqual(alphas.size(), (5, 10))
        self.assertTrue(np.allclose(alphas.sum(1).detach().numpy(), np.ones_like(alphas.sum(1).detach().numpy())))
        self.assertEqual(summaries.size(), (5, 7))

    def test_fwdmul_attention(self):
        x = torch.randn(5, 10, 7)
        y = torch.randn(5, 7)

        a = q.FwdAttention(7, 7, 7)

        alphas, summaries = a(y, x)
        print(alphas.size())
        # test shapes
        self.assertEqual(alphas.size(), (5, 10))
        self.assertTrue(np.allclose(alphas.sum(1).detach().numpy(), np.ones_like(alphas.sum(1).detach().numpy())))
        self.assertEqual(summaries.size(), (5, 7))


class TestAttentionBases(TestCase):
    def test_it(self):
        class DotAttentionMem(q.Attention, q.DotAttention):
            pass
        a = DotAttentionMem()

        x = torch.randn(5, 10, 7)
        y = torch.randn(5, 7)

        alphas, summaries = a(y, x)
        print(alphas.size())
        # test shapes
        self.assertEqual(alphas.size(), (5, 10))
        self.assertTrue(np.allclose(alphas.sum(1).detach().numpy(), np.ones_like(alphas.sum(1).detach().numpy())))
        self.assertEqual(summaries.size(), (5, 7))

    def test_it_with_coverage(self):
        class DotAttentionWithCov(q.AttentionWithCoverage, q.DotAttention):
            pass

        a = DotAttentionWithCov()
        x = torch.randn(5, 10, 7)
        y = torch.randn(5, 7)

        alphas, summaries = a(y, x)
        print(alphas.size())
        # test shapes
        self.assertEqual(alphas.size(), (5, 10))
        self.assertTrue(np.allclose(alphas.sum(1).detach().numpy(), np.ones_like(alphas.sum(1).detach().numpy())))
        self.assertEqual(summaries.size(), (5, 7))

# endregion


# region decoders
class TestDecoders(TestCase):
    def test_tf_decoder(self):
        decodercell = torch.nn.Linear(7, 12)
        x = torch.randn(5, 10, 7)
        decoder = q.TFDecoder(decodercell)
        y = decoder(x)
        self.assertEqual(y.size(), (5, 10, 12))

    def test_free_decoder(self):
        decodercell = torch.nn.Sequential(torch.nn.Embedding(12, 7),
                                          torch.nn.Linear(7, 12))
        x = torch.randint(0, 7, (5,), dtype=torch.int64)
        decoder = q.FreeDecoder(decodercell, maxtime=10)
        y = decoder(x)
        self.assertEqual(y.size(), (5, 10, 12))


class TestDecoderCell(TestCase):
    def test_it(self):
        x = np.random.randint(0, 100, (1000, 7))
        y_inp = x[:, :-1]
        y_out = x[:, 1:]
        wD = dict((chr(xi), xi) for xi in range(100))

        ctx = torch.randn(1000, 8, 30)

        decoder_emb = q.WordEmb(20, worddic=wD)
        decoder_lstm = q.LSTMCell(20, 30)
        decoder_att = q.DotAttention()
        decoder_out = q.WordLinout(60, worddic=wD)

        decoder_cell = q.DecoderCell(decoder_emb, decoder_lstm, decoder_att, None, decoder_out)
        decoder_tf = q.TFDecoder(decoder_cell)

        y = decoder_tf(torch.tensor(x), ctx=ctx)

        self.assertTrue(y.size(), (1000, 7, 100))




# endregion