import math
import tqdm
import torch
import gpytorch
import numpy as np


def _rbf_kernel_diff(dist_mat, os=1.):
    kernel = dist_mat.mul(-0.5).exp().mul(os)
    return kernel


def rbf_kernel(x1, x2, ls=1., os=1.):
    x1 = x1 / ls
    x2 = x2 / ls
    dist_mat = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(2.).sum(dim=-1)
    return _rbf_kernel_diff(dist_mat, os=os)


def _additive_rbf_kernel_diff(dist_mats, os=1.):
    kernel = dist_mats.mul(-0.5).exp().mul(os)
    kernel = kernel.mean(dim=-3)
    return kernel


def limiting_kernel(kern, os1=1., ls2=1., os2=1.):
    kern = (os2) * ls2 / (os1 - kern).mul(2.).add(ls2 ** 2).sqrt()
    return kern


class Limit3L_DGP(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, num_locs=11, **kwargs):
        super().__init__(has_lengthscale=False)
        self.num_locs = num_locs
        self.width = 2

        # Hypers
        self.register_parameter("raw_ls1", torch.nn.Parameter(torch.tensor(-0.0311)))
        self.register_parameter("raw_os1", torch.nn.Parameter(torch.tensor(1.39)))
        self.register_parameter("raw_ls2", torch.nn.Parameter(torch.tensor(-1.72)))
        self.register_parameter("raw_os2", torch.nn.Parameter(torch.tensor(0.2329)))
        self.register_parameter("raw_ls3", torch.nn.Parameter(torch.tensor(-1.72)))
        self.register_parameter("raw_os3", torch.nn.Parameter(torch.tensor(0.2329)))

        # Quadrature
        locations, weights = np.polynomial.hermite.hermgauss(num_locs)
        self.register_buffer("quad_locs", torch.tensor(locations).to(self.raw_ls1.dtype))
        self.register_buffer("quad_weights", torch.tensor(weights).to(self.raw_ls1.dtype))

    @property
    def ls1(self):
        return torch.nn.functional.softplus(self.raw_ls1)

    @property
    def ls2(self):
        return torch.nn.functional.softplus(self.raw_ls2)

    @property
    def ls3(self):
        return torch.nn.functional.softplus(self.raw_ls3)

    @property
    def os1(self):
        return torch.nn.functional.softplus(self.raw_os1)

    @property
    def os2(self):
        return torch.nn.functional.softplus(self.raw_os2)

    @property
    def os3(self):
        return torch.nn.functional.softplus(self.raw_os3)

    def forward(self, x1, x2, diag=False, **params):
        ls1 = self.ls1
        ls2 = self.ls2
        ls3 = self.ls3
        os1 = self.os1
        os2 = self.os2
        os3 = self.os3

        # Get quadrature locations
        kern = rbf_kernel(x1=x1, x2=x2, ls=ls1, os=os1)
        kern = limiting_kernel(kern=kern, os1=os1, ls2=ls2, os2=os2)
        kern = limiting_kernel(kern=kern, os1=os2, ls2=ls3, os2=os3)
        return kern


class ExactGP3L(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = Limit3L_DGP()
        self.likelihood.raw_noise.data.fill_(-1.5)

    @property
    def ls1(self):
        return self.covar_module.ls1

    @property
    def ls2(self):
        return self.covar_module.ls2

    @property
    def ls3(self):
        return self.covar_module.ls3

    @property
    def os1(self):
        return self.covar_module.os1

    @property
    def os2(self):
        return self.covar_module.os2

    @property
    def os3(self):
        return self.covar_module.os3

    @property
    def noise(self):
        return self.likelihood.noise.squeeze()

    def forward(self, x):
        mean = self.mean_module(x)
        covar = gpytorch.lazify(self.covar_module(x).evaluate())
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def evaluate(self, train_x, train_y, test_x, test_y, savedir):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(1e6):
            self.train()
            nmll = -mll(self(train_x), train_y)

            self.eval()
            pred_dist = self.likelihood(self(test_x))
            nll = -pred_dist.to_data_independent_dist().log_prob(test_y).mean(-1)
            rmse = (pred_dist.mean - test_y).pow(2).mean(-1).sqrt()

        return nmll.item(), nll.item(), rmse.item()

    def kernel_fit(self, train_x, train_y, savedir):
        self.eval()
        with torch.no_grad(), gpytorch.settings.prior_mode(True):
            prior_dist = self.likelihood(self(train_x))
            log_probs = prior_dist.log_prob(train_y)
        return log_probs

    def optimize(self, train_x, train_y, savedir, num_iter=100, lr=0.01):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        iterator = tqdm.tqdm(range(num_iter), desc="ExactGP Training")

        with gpytorch.settings.max_cholesky_size(2e3):
            for _ in iterator:
                optimizer.zero_grad()
                output = self(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
                iterator.set_postfix(
                    loss=loss.item(),
                    ls1=self.ls1.item(),
                    ls2=self.ls2.item(),
                    ls3=self.ls3.item(),
                    os1=self.os1.item(),
                    os2=self.os2.item(),
                    os3=self.os3.item(),
                    noise=self.noise.item(),
                )

        return self
