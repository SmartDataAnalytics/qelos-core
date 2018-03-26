import qelos_core as q
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms


def run(lr=0.01, epochs=10, batsize=64, momentum=0.5, cuda=False, gpu=0, seed=1):
    settings = locals().copy()
    logger = q.Logger(prefix="mnist")
    logger.save_settings(**settings)

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.set_device(gpu)
        torch.cuda.manual_seed(seed)

    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batsize, shuffle=False, **kwargs)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    model = Net()

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    trainer = q.trainer(model).on(train_loader)\
        .loss(torch.nn.NLLLoss(), q.Accuracy())\
        .optimizer(optim).cuda(cuda)
    validator = q.tester(model).on(test_loader)\
        .loss(torch.nn.NLLLoss(), q.Accuracy())\
        .cuda(cuda)

    logger.loglosses(trainer, "train.losses")
    logger.loglosses(validator, "valid.losses")

    q.train(trainer, validator).run(epochs)


if __name__ == "__main__":
    q.argprun(run)