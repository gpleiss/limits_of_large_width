import logging
import torch
import os
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from torchvision.datasets import MNIST, CIFAR10
from math import ceil, floor
import numpy as np
import pandas as pd


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_toy_data(data_dir, w, d):
    data = torch.load(os.path.join(data_dir, f"dgp_toy_d{d}.pth"))
    X = data["x"]
    y = data["y"][w]

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    train_n = int(floor(0.5 * X.size(0)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()

    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()

    logging.info("Loaded data with input dimension of {}".format(test_x.size(-1)))

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    return train_x, train_y, test_x, test_y, test_x, test_y


def load_uci_data(data_dir, dataset, seed, n=None):
    set_seed(seed)

    if dataset == "energy":
        data = torch.tensor(pd.read_excel(os.path.join(data_dir, dataset + '.xlsx')).values[:, :-3])
    elif dataset == "power":
        data = torch.tensor(pd.read_excel(os.path.join(data_dir, dataset + '.xlsx')).values)
    elif dataset == "concrete":
        data = torch.tensor(pd.read_excel(os.path.join(data_dir, dataset + '.xls')).values)
    elif dataset == "boston":
        data = torch.tensor(pd.read_fwf(os.path.join(data_dir, dataset + '.data')).values)
    else:
        data = torch.tensor(loadmat(os.path.join(data_dir, dataset + '.mat'))['data'])

    data = data.float()
    X = data[:, :-1]

    good_dimensions = X.var(dim=-2) > 1.0e-10
    if int(good_dimensions.sum()) < X.size(1):
        logging.info("Removed %d dimensions with no variance" % (X.size(1) - int(good_dimensions.sum())))
        X = X[:, good_dimensions]

    if dataset in ['keggundirected', 'slice']:
        X = torch.Tensor(SimpleImputer(missing_values=np.nan).fit_transform(X.data.numpy()))

    y = data[:, -1]

    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    y -= y.mean()
    y /= y.std()

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    train_n = int(floor(0.75 * X.size(0)))
    valid_n = int(floor(0.10 * X.size(0)))

    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()

    if n is not None:
        train_x = train_x[:n]
        train_y = train_y[:n]

    valid_x = X[train_n:train_n+valid_n, :].contiguous()
    valid_y = y[train_n:train_n+valid_n].contiguous()

    test_x = X[train_n+valid_n:, :].contiguous()
    test_y = y[train_n+valid_n:].contiguous()

    logging.info("Loaded data with input dimension of {}".format(test_x.size(-1)))

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        valid_x = valid_x.cuda()
        valid_y = valid_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    return train_x, train_y, test_x, test_y, valid_x, valid_y


def load_mnist(data_dir, dataset, seed, n=None):
    set_seed(seed) 

    train_set = MNIST(data_dir, train=True, download=True)
    test_set = MNIST(data_dir, train=False, download=True)
    
    y = train_set.targets
    X = train_set.data.view(y.size(0), -1).float()
    x_mean = X.mean()
    x_stdv = X.std().clamp_min_(1e-3)
    X = X.sub_(x_mean).div_(x_stdv)

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]
    
    train_x = X[:-10000]
    train_y = y[:-10000]
    valid_x = X[-10000:]
    valid_y = y[-10000:]

    if n is not None:
        train_x = train_x[:n]
        train_y = train_y[:n]

    test_y = test_set.targets
    test_x = test_set.data.view(test_y.size(0), -1).float().sub_(x_mean).div_(x_stdv)

    logging.info("Loaded data with input dimension of {}".format(test_x.size(-1)))

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        valid_x = valid_x.cuda()
        valid_y = valid_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    return train_x, train_y, test_x, test_y, valid_x, valid_y


def load_cifar10(data_dir, dataset, seed, n=None):
    set_seed(seed) 

    train_set = CIFAR10(data_dir, train=True, download=True)
    test_set = CIFAR10(data_dir, train=False, download=True)
    
    y = torch.tensor(train_set.targets)
    X = torch.tensor(train_set.data).float().permute(0, 3, 1, 2)
    x_mean = X.mean(dim=[0, -2, -1], keepdim=True)
    x_stdv = X.std(dim=[0, -2, -1], keepdim=True).clamp_min_(1e-3)
    X = X.sub_(x_mean).div_(x_stdv)
    X = X.contiguous().view(X.size(0), -1)

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]
    
    train_x = X
    train_y = y

    if n is not None:
        train_x = train_x[:n]
        train_y = train_y[:n]

    test_y = torch.tensor(test_set.targets)
    test_x = torch.tensor(test_set.data).float().permute(0, 3, 1, 2).sub_(x_mean).div_(x_stdv)
    test_x = test_x.contiguous().view(test_x.size(0), -1)

    logging.info("Loaded data with input dimension of {}".format(test_x.size(-1)))
    return train_x, train_y, test_x, test_y, test_x, test_y
