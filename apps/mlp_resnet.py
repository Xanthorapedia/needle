import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm, drop_prob=0.1):
    np.random.seed(4)
    fn = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    block = nn.Sequential(
        nn.Residual(fn),
        nn.ReLU()
    )
    return block


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm, drop_prob=0.1):
    np.random.seed(4)
    modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    modules += [ResidualBlock(hidden_dim, hidden_dim // 2, norm) for _ in range(num_blocks)]
    modules += [nn.Linear(hidden_dim, num_classes)]
    return nn.Sequential(*modules)


def train_epoch(dataloader, model, loss_fn, opt):
    np.random.seed(4)
    model.train()
    losses, accs = [], []
    for _, (batch_x, batch_y) in enumerate(dataloader):
        opt.reset_grad()
        out = model(batch_x)
        loss = loss_fn(out, batch_y)
        loss.backward()
        opt.step()

        losses.append([loss.numpy()] * len(batch_x.numpy()))
        pred_y = np.argmax(out.numpy(), axis=-1)
        accs.append(pred_y == batch_y.numpy())

    return np.concatenate(accs).mean(), np.concatenate(losses).mean()



def evaluate(dataloader, model, loss_fn=nn.SoftmaxLoss()):
    np.random.seed(4)
    model.eval()
    losses, accs = [], []
    for _, (batch_x, batch_y) in enumerate(dataloader):
        out = model(batch_x)
        loss = loss_fn(out, batch_y)
        losses.append([loss.numpy()] * len(batch_x.numpy()))
        pred_y = np.argmax(out.numpy(), axis=-1)
        accs.append(pred_y == batch_y.numpy())
    return np.concatenate(accs).mean(), np.concatenate(losses).mean()


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    model = MLPResNet(784, hidden_dim)

    ########## Train ##########
    train_dataset = ndl.data.MNISTDataset(\
            os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
            os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))

    train_dataloader = ndl.data.DataLoader(\
             dataset=train_dataset,
             batch_size=batch_size,
             shuffle=True,
             collate_fn=ndl.data.collate_mnist)

    loss_func = nn.SoftmaxLoss()
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_acc, train_loss = train_epoch(train_dataloader, model, loss_func, opt)

    ########## Test ##########
    test_dataset = ndl.data.MNISTDataset(\
            os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
            os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    test_dataloader = ndl.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size,
             collate_fn=ndl.data.collate_mnist)
    
    test_acc, test_loss = evaluate(test_dataloader, model, loss_func)

    return train_acc, train_loss, test_acc, test_loss


def num_params(model):
    return np.sum([np.prod(x.shape) for x in model.parameters()])

if __name__ == "__main__":
    train_mnist(data_dir="../data")
