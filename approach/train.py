from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
import random
import os
import time
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from dataset import TorchDataset
from model import Model
import tools as utils


def parse_args():
    """ Parse arguments from terminal. """
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size of dataloader')
    parser.add_argument('--epochs', type=int, default=700,
                    help='Epochs of training')
    parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate of optimizer')

    parser.add_argument('--backup', type=str, default='backup',
                    help='Directory for saving checkpoints')
    return parser.parse_args()


def set_seed(seed):
    """ Set seed to determine random behavior. """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_device():
    """ Set device used for training, only support 1-GPU. """
    if torch.cuda.is_available():
        print('cuda is available')
        return torch.device('cuda:0')
    else:
        print('cuda is not available')
        return torch.device('cpu')


def save(model, optimizer, epoch, to_dir):
    """ Save training state to local specified by <to_dir>. """
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    path = os.path.join(to_dir, f'ckpt_{epoch}.pkl')
    torch.save(ckpt, path)
    print(f'save checkpoint to {path} in epoch {epoch}')


def test(model, loader):
    model.eval()
    outputs = []
    targets = []
    for x, y in loader:
        x = x.to(dev)
        y = y.to(dev)

        output = model(x)
        output = np.argmax(output.detach().cpu().numpy(), axis=1).tolist()
        target = y.detach().cpu().numpy().tolist()

        outputs.extend(output)
        targets.extend(target)

    # return str(classification_report(targets, outputs))
    return accuracy_score(targets, outputs)



def train(model, train_loader, loss_func, optim, dev):
    total_loss = 0
    model.train()

    for x, y in tqdm(train_loader):
        x = x.to(dev)
        y = y.to(dev)

        output = model(x)
        loss = loss_func(output, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        total_loss += loss.item()

    return total_loss / len(train_loader)


if __name__ == '__main__':
    args = parse_args()
    print(f'arguments are: \n'
          f'{utils.format_args(args)}')

    # setup random seed and device
    set_seed(args.seed)
    dev = set_device()

    # read dataset
    train_set = TorchDataset(args.index, train=True)
    test_set = TorchDataset(args.index, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    # setup model, loss and optimizer
    model = Model(input_dims=100).to(dev)
    adam = torch.optim.Adam(model.parameters(), args.lr)
    loss_func = nn.CrossEntropyLoss().to(dev)

    for e in range(args.epochs):

        avg_loss = train(model, train_loader, loss_func, adam, dev)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        print(f'epochs: {e + 1}, loss: {avg_loss}, train_acc: {train_acc}, test_acc: {test_acc}')

        # save
        if (e + 1) % 30 == 0:
            save(model, adam, e + 1, args.backup)
