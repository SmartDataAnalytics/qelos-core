import unittest
from unittest import TestCase
import torchvision.datasets as dset
import torchvision.transforms as transforms
import qelos_core as q
import torch


class MyTestCase(TestCase):
    def test_something(self):
        self.assertEqual(True, True)


def tst_inception_cifar10(self):
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)
    cifar = dset.CIFAR10(root='data/', download=True,
                         transform=transforms.Compose([
                             transforms.Scale(32),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])
                         )

    cifar = IgnoreLabelDataset(cifar)
    cifar_loader = q.dataload(cifar, batch_size=32)
    inception_scorer = q.gan.IS()

    print ("Calculating Inception Score...")

    scores = inception_scorer.get_scores(cifar_loader)
    print(scores)


if __name__ == '__main__':
    tst_inception_cifar10()
    unittest.main()
