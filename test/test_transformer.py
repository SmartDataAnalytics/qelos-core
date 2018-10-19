from unittest import TestCase
import torch
from qelos_core.transformer import *
from qelos_core.transformer_huggingface import Attention as HF_Attention
import numpy as np


class TestAttention(TestCase):
    def test_it(self):
        x = torch.randn(2, 3, 12)
        x_ref = x + 0.
        x.requires_grad = True
        x_ref.requires_grad = True
        numheads = 6
        m = MultiHeadAttention(12, numheads=numheads, bidir=False)
        m_ref = HF_Attention(12, 3, cfg_n_head=numheads)
        # m.qkv_proj = m_ref.c_attn
        # m.vw_proj = m_ref.c_proj
        m_ref.c_attn.w = torch.nn.Parameter(
            torch.cat([m.q_proj.weight.t(),
                       m.k_proj.weight.t(),
                       m.v_proj.weight.t()], 1)
        )
        m_ref.c_proj.w = torch.nn.Parameter(m.vw_proj.weight.t())
        m_ref.c_attn.b = torch.nn.Parameter(
            torch.cat([m.q_proj.bias,
                       m.k_proj.bias,
                       m.v_proj.bias], 0)
        )
        m_ref.c_proj.b = m.vw_proj.bias

        y = m(x)
        y_ref = m_ref(x_ref)

        # print(_x.size(), _x_ref.size())
        # print((_x - _x_ref).norm())
        #
        # print(vw.size(), vw_ref.size())
        # print((vw - vw_ref).norm())

        # y, _x, q, k, v, w_a, w_b = m(x)
        # y_ref, _x_ref, _q, _k, _v, _w_a, _w_b = m_ref(x)
        #
        # print((_x - _x_ref).norm())
        # print((q - _q.transpose(1, 2)).norm())
        # print((k - _k.permute(0, 3, 1, 2)).norm())
        # print((v - _v.transpose(1, 2)).norm())
        # print((w_a - _w_a).norm())
        # print((w_b - _w_b).norm())
        print(y.size(), y_ref.size())
        print((y - y_ref).norm())
        self.assertTrue((np.allclose(y.detach().numpy(), y_ref.detach().numpy())))
        print("outputs good")

        y.sum().backward()
        y_ref.sum().backward()

        print((x.grad - x_ref.grad).norm())
        self.assertTrue((np.allclose(x.grad.detach().numpy(), x_ref.grad.detach().numpy(), atol=1e-5)))
        print("grads good")

    def test_mask(self):
        x = torch.randn(4, 4, 12)
        x.requires_grad = True
        numheads = 6
        m = MultiHeadAttention(12, numheads=numheads)
        # mask = None
        mask = torch.tensor([[1,1,1,0],[1,0,0,0],[1,1,1,1],[1,0,1,0]])

        y = m(x, mask=mask)
        print(y.size())
        l = y.sum()
        l.backward()
        print(x.grad[0, :, :].norm(1, dim=1))
        print(x.grad.norm(1, dim=2))


class TestAttentionCell(TestCase):
    def test_it(self):
        x = torch.randn(4, 5, 12)
        x.requires_grad = True
        numheads = 6
        m = MultiHeadAttention(12, numheads=numheads, bidir=False)
        mc = MultiHeadAttentionCell(m, 3)

        ys = []
        for i in range(x.size(1)):
            y = mc(x[:, i:i+1])
            print(y.size())
            ys.append(y)



class TestEncoderBlock(TestCase):
    def test_it(self):
        indim = 10
        kdim = 12
        vdim = 15
        numheads = 3
        m = EncoderBlock(indim, kdim=kdim, vdim=vdim, bidir=True,
                         numheads=numheads, activation=GeLU)

        batsize = 4
        seqlen = 5

        x = torch.randn(batsize, seqlen, indim)
        x.requires_grad = True
        xmask = torch.ones_like(x[:, :, 0]).byte()
        xmask[:, -1] = 0
        xmask[0, -1] = 1
        xmask[1, -2] = 0
        print(xmask)

        #
        # y = m(x)
        y = m(x, mask=xmask)

        print(y.size())
        l = y.norm(1)
        l.backward()

        print(x.grad.norm(1, 2))


