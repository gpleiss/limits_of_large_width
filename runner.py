import fire
import tqdm
import logging
import os
import sys
import math
import json
import datetime
import torch
import gpytorch
import random
import pandas as pd
from datasets import load_uci_data, load_mnist, load_cifar10, load_toy_data
from exact_gp import ExactGP
from exact_gp_3l import ExactGP3L
from exact_gp_acos import ExactGP_Acos
from exact_gp_acos3 import ExactGP_Acos3
from deep_gp_svi import DeepGP_SVI
from deep_gp_hmc import DeepGP_HMC
from deep_gp_3l2l_hmc import DeepGP3L2L_HMC
from deep_gp_3l3l_hmc import DeepGP3L3L_HMC
from nn import NN
from resnet import ResNet
from lenet import LeNet
from nn_hmc import NN_HMC
from nn_3l_hmc import NN3L_HMC


class Runner(object):
    """
    Main module for running experiments. Can call `load`, `save`, `train`, `test`, etc.
    """
    def __init__(self, data, save, device, seed, dataset, n, model, **model_args):
        self.data = data
        self.device = device
        self.savedir = save
        self.seed = seed
        self.n = n

        # Seed
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)

        # Set GPU
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

        # Logging
        self.timestring = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logging.basicConfig(
            format="%(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(self.savedir, "log-%s.log" % self.timestring)),
            ],
            level=logging.INFO,
        )
        logging.info("Data dir:\t%s" % data)
        logging.info("Save dir:\t%s\n" % save)

        # Make dataset
        self.dataset = dataset
        if dataset == "mnist":
            self.train_x, self.train_y, self.test_x, self.test_y, self.valid_x, self.valid_y = load_mnist(
                data_dir=self.data, dataset=self.dataset, seed=self.seed, n=self.n
            )
            model_args["classification"] = True
            model_args["num_outputs"] = 10
        elif dataset == "cifar10":
            self.train_x, self.train_y, self.test_x, self.test_y, self.valid_x, self.valid_y = load_cifar10(
                data_dir=self.data, dataset=self.dataset, seed=self.seed, n=self.n
            )
            model_args["classification"] = True
            model_args["num_outputs"] = 10
        elif "toy" in dataset:
            # e.g. toyw2d4
            w = int(dataset.split("w")[1].split("d")[0])
            d = int(dataset.split("d")[1])
            self.train_x, self.train_y, self.test_x, self.test_y, self.valid_x, self.valid_y = load_toy_data(
                data_dir=self.data, w=w, d=d,
            )
        else:
            self.train_x, self.train_y, self.test_x, self.test_y, self.valid_x, self.valid_y = load_uci_data(
                data_dir=self.data, dataset=self.dataset, seed=self.seed, n=self.n
            )
        logging.info(f"Dataset: {self.dataset}")
        logging.info(f"Num train: {len(self.train_x)}")
        logging.info(f"Num valid: {len(self.valid_x)}")
        logging.info(f"Num test: {len(self.test_x)}")

        # Make model
        if model == "gp":
            self.model = ExactGP(self.train_x, self.train_y, **model_args)
        elif model == "gp_3l":
            self.model = ExactGP3L(self.train_x, self.train_y, **model_args)
        elif model == "gp_acos":
            self.model = ExactGP_Acos(self.train_x, self.train_y, **model_args)
        elif model == "gp_acos3":
            self.model = ExactGP_Acos3(self.train_x, self.train_y, **model_args)
        elif model == "deep_gp_svi":
            self.model = DeepGP_SVI(self.train_x, self.train_y, **model_args)
        elif model == "deep_gp_hmc":
            self.model = DeepGP_HMC(self.train_x, self.train_y, **model_args)
        elif model == "deep_gp_3l2l_hmc":
            self.model = DeepGP3L2L_HMC(self.train_x, self.train_y, **model_args)
        elif model == "deep_gp_3l3l_hmc":
            self.model = DeepGP3L3L_HMC(self.train_x, self.train_y, **model_args)
        elif model == "nn":
            self.model = NN(self.train_x, self.train_y, **model_args)
        elif model == "resnet":
            self.model = ResNet(self.train_x, self.train_y, **model_args)
        elif model == "lenet":
            self.model = LeNet(self.train_x, self.train_y, **model_args)
        elif model == "nn_hmc":
            self.model = NN_HMC(self.train_x, self.train_y, **model_args)
        elif model == "nn_3l_hmc":
            self.model = NN3L_HMC(self.train_x, self.train_y, **model_args)
        else:
            raise ValueError(f"Unknown model {model}.")
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Log model
        logging.info(f"Initializing...")
        logging.info(f"Model: {model}")
        logging.info(f"Model args:")
        for key, val in model_args.items():
            logging.info(f" - {key}: {val}")
        logging.info(self.model)
        logging.info("")

    def done(self):
        """Break out of the runner"""
        return None

    def initialize(self, other):
        other_model = load(other, data=self.data).model
        self.model.initialize_from_previous_model(other_model, self.train_x, self.savedir)
        return self

    def load(self, save=None, suffix=""):
        """
        Load a previously saved model state dict.

        :param str save: (optional) Which folder to load the saved model from.
            Will default to the current runner's save dir.
        :param str suffix: (optional) Which model file to load (e.g. "model.pth.last").
            By default will load "model.pth" which contains the early-stopped model.
        """
        save = save or self.savedir
        state_dict = torch.load(os.path.join(save, f"model.pth{suffix}"), map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=False)
        return self

    def save(self, save=None, suffix=""):
        """
        Save the current state dict

        :param str save: (optional) Which folder to save the model to.
            Will default to the current runner's save dir.
        :param str suffix: (optional) A suffix to append to the save name.
        """
        save = save or self.savedir
        torch.save(self.model.state_dict(), os.path.join(save, f"model.pth{suffix}"))
        return self

    def kernel_fit(self):
        log_probs = self.model.kernel_fit(self.train_x, self.train_y, self.savedir)
        logging.info(f"Log probs (min/mean/max): {log_probs.min().item()}, {log_probs.mean().item()}, {log_probs.max().item()}")
        torch.save(log_probs, os.path.join(self.savedir, "log_probs.pth"))
        return self

    def test(self):
        nmll, nll, err = self.model.evaluate(self.train_x, self.train_y, self.test_x, self.test_y, self.savedir)
        logging.info(f"Negative MLL: {nmll}")
        if self.dataset == "mnist":
            logging.info(f"Predictive Err: {err}")
        else:
            logging.info(f"Predictive RMSE: {err}")
        logging.info(f"Predictive NLL: {nll}")
        return self

    def train(self, **train_args):
        self.model.optimize(self.train_x, self.train_y, self.savedir, **train_args)
        self.save()
        return self


def new(save, overwrite=False, **kwargs):
    kwargs["seed"] = kwargs.get("seed", 0)
    kwargs["device"] = kwargs.get("device", 0)
    kwargs["n"] = kwargs.get("n", None)
    kwargs["data"] = kwargs.get("data", os.path.join(os.getenv("HOME"), "datasets", "uci"))
    overwrite = overwrite or ("results/test" in save)

    if os.path.exists(save):
        if not overwrite:
            raise ValueError(f"{save} already exists. Set overwrite=True to overwrite.")
    else:
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception(f"{save} is not a dir")

    # Write args to JSON
    with open(os.path.join(save, "args.json"), "w") as f:
        json.dump(kwargs, f, indent=4)
    return Runner(save=save, **kwargs)


def load(save, data=None):
    if not os.path.isdir(save):
        raise Exception(f"{save} is not a dir")

    # Read args from JSON
    with open(os.path.join(save, "args.json"), "r") as f:
        kwargs = json.load(f)
    if data is not None:
        kwargs["data"] = data
    return Runner(save=save, **kwargs).load()


if __name__ == "__main__":
    fire.Fire()
