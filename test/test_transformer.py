from unittest import TestCase
import torch
from qelos_core.transformer import *
from qelos_core.transformer_huggingface import Attention as HF_Attention


class TestConv1D(TestCase):
    def test_it(self):
        x = torch.randn(2, 3, 4)
        x.requires_grad = True
        m = Conv1D(4, 5)
        y = m(x)
        print(y.size())

        l = y[0, 0].sum()
        l.backward()

        print(x.grad[:, :, 0])


class TestAttention(TestCase):
    def test_it(self):
        x = torch.randn(2, 3, 12)
        x_ref = x + 0.
        x.requires_grad = True
        x_ref.requires_grad = True
        numheads = 6
        m = MultiHeadAttention(12, numheads=numheads)
        m_ref = HF_Attention(12, 3, cfg_n_head=numheads)
        # m.qkv_proj = m_ref.c_attn
        # m.vw_proj = m_ref.c_proj
        m_ref.c_attn.w = torch.nn.Parameter(m.qkv_proj.layer.weight.t())
        m_ref.c_proj.w = torch.nn.Parameter(m.vw_proj.layer.weight.t())
        m_ref.c_attn.b = m.qkv_proj.layer.bias
        m_ref.c_proj.b = m.vw_proj.layer.bias

        y = m(x)
        y_ref = m_ref(x_ref)
        #
        # y, _x, q, k, v, w_a, w_b = m(x)
        # y_ref, _x_ref, _q, _k, _v, _w_a, _w_b = m_ref(x)
        #
        # print((_x - _x_ref).norm())
        # print((q - _q.transpose(1, 2)).norm())
        # print((k - _k.permute(0, 3, 1, 2)).norm())
        # print((v - _v.transpose(1, 2)).norm())
        # print((w_a - _w_a).norm())
        # print((w_b - _w_b).norm())
        print((y - y_ref).norm())
        self.assertTrue((y - y_ref).norm().item() == 0)
        print("outputs good")

        y.sum().backward()
        y_ref.sum().backward()

        print((x.grad - x_ref.grad).norm())
        self.assertTrue((x.grad - x_ref.grad).norm().item() == 0)
        print("grads good")


