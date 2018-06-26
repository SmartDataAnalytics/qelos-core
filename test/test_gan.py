import unittest
from unittest import TestCase
import torchvision.datasets as dset
import torchvision.transforms as transforms
import qelos_core as q
import torch


class MyTestCase(TestCase):
    def test_something(self):
        self.assertEqual(True, True)


def tst_inception_cifar10(cuda=False, gpu=1, batsize=32):
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
    device = torch.device("cpu") if not cuda else torch.device("cuda", gpu)
    print(device, cuda)
    cifar = IgnoreLabelDataset(cifar)
    cifar_loader = q.dataload(cifar, batch_size=batsize)
    inception_scorer = q.gan.IS(device=device)
    fid_scorer = q.gan.FID(inception=inception_scorer.inception, device=device)

    print ("Calculating Inception Score...")

    scores = inception_scorer.get_scores(cifar_loader)
    print(scores)

    print("Calculating FIDs...")
    fids = fid_scorer.get_distance(cifar_loader, cifar_loader)
    print(fids)



if __name__ == '__main__':
    q.argprun(tst_inception_cifar10)

