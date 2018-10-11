import qelos_core as q
import torch
import numpy as np


def run(lr=0.001):
    x = np.random.random((1000, 5)).astype("float32")
    y = np.random.randint(0, 5, (1000,)).astype("int64")

    trainloader = q.dataload(x[:800], y[:800], batch_size=100)
    validloader = q.dataload(x[800:], y[800:], batch_size=100)

    m = torch.nn.Sequential(torch.nn.Linear(5, 100),
                            torch.nn.Linear(100, 5))

    m[1].weight.requires_grad = False

    params = m.parameters()
    for param in params:
        print(param.requires_grad)

    init_val = m[1].weight.detach().numpy()

    optim = torch.optim.Adam(q.params_of(m), lr=lr)

    trainer = q.trainer(m).on(trainloader).loss(torch.nn.CrossEntropyLoss()).optimizer(optim).epochs(100)

    # for b, (i, e) in trainer.inf_batches():
    #     print(i, e)

    validator = q.tester(m).on(validloader).loss(torch.nn.CrossEntropyLoss())

    q.train(trainer, validator).run()

    new_val = m[1].weight.detach().numpy()

    print(np.linalg.norm(new_val - init_val))


if __name__ == "__main__":
    run()