import tqdm
import math
import torch
import gpytorch


def protect_from_nans(grad):
    grad = torch.where(torch.isnan(grad), grad.new_zeros([]), grad)
    return grad


class Acos3Kernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, **kwargs):
        super().__init__(has_lengthscale=False)
        self.register_parameter("raw_sigma_w1", torch.nn.Parameter(torch.tensor(0.)))
        self.register_parameter("raw_sigma_b1", torch.nn.Parameter(torch.tensor(0.)))
        self.register_parameter("raw_sigma_w2", torch.nn.Parameter(torch.tensor(0.)))
        self.register_parameter("raw_sigma_b2", torch.nn.Parameter(torch.tensor(0.)))
        self.register_parameter("raw_sigma_w3", torch.nn.Parameter(torch.tensor(0.)))
        self.register_parameter("raw_sigma_b3", torch.nn.Parameter(torch.tensor(0.)))

    @property
    def sigma_w1(self):
        return torch.nn.functional.softplus(self.raw_sigma_w1)

    @property
    def sigma_b1(self):
        return torch.nn.functional.softplus(self.raw_sigma_b1)

    @property
    def sigma_w2(self):
        return torch.nn.functional.softplus(self.raw_sigma_w2)

    @property
    def sigma_b2(self):
        return torch.nn.functional.softplus(self.raw_sigma_b2)

    @property
    def sigma_w3(self):
        return torch.nn.functional.softplus(self.raw_sigma_w3)

    @property
    def sigma_b3(self):
        return torch.nn.functional.softplus(self.raw_sigma_b3)

    def _forward2(self, x1, x2, diag=False, **params):
        sigma_w1_2 = self.sigma_w1 ** 2
        sigma_w2_2 = self.sigma_w2 ** 2
        sigma_b1_2 = self.sigma_b1 ** 2
        sigma_b2_2 = self.sigma_b2 ** 2

        if diag:
            k1 = sigma_b1_2 + sigma_w1_2 * ( x1 * x2 ).sum(dim=-1)
            norms = k1
        else:
            k1 = sigma_b1_2 + sigma_w1_2 * ( x1 @ x2.transpose(-1, -2) )
            k1_x1 = sigma_b1_2 + sigma_w1_2 * x1.norm(dim=-1).pow(2.)
            k1_x2 = sigma_b1_2 + sigma_w1_2 * x2.norm(dim=-1).pow(2.)
            norms = torch.sqrt(k1_x1.unsqueeze(-1) @ k1_x2.unsqueeze(-2))

        cos_theta = (k1 / norms).clamp(-1., 1.)
        theta = torch.acos(cos_theta)
        if cos_theta.requires_grad:
            cos_theta.register_hook(protect_from_nans)
            
        res = sigma_b2_2 + sigma_w2_2 / (2 * math.pi) * norms * (
            torch.sin(theta) + (math.pi - theta) * cos_theta
        )
        return res

    def forward(self, x1, x2, diag=False, **params):
        sigma_w3_2 = self.sigma_w3 ** 2
        sigma_b3_2 = self.sigma_b3 ** 2

        k2 = self._forward2(x1, x2, diag=diag, **params)
        k2_x1 = self._forward2(x1, x1, diag=True, **params)
        k2_x2 = self._forward2(x2, x2, diag=True, **params)

        norms = torch.sqrt(k2_x1.unsqueeze(-1) @ k2_x2.unsqueeze(-2))
        cos_theta = (k2 / norms).clamp(-1., 1.)
        theta = torch.acos(cos_theta)
        if cos_theta.requires_grad:
            cos_theta.register_hook(protect_from_nans)
            
        res = sigma_b3_2 + sigma_w3_2 / (2 * math.pi) * norms * (
            torch.sin(theta) + (math.pi - theta) * cos_theta
        )

        return res


class ExactGP_Acos3(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = Acos3Kernel()
        self.likelihood.raw_noise.data.fill_(-1.5)

    @property
    def ls1(self):
        return self.covar_module.sigma_b1

    @property
    def ls2(self):
        return self.covar_module.sigma_b2

    @property
    def ls3(self):
        return self.covar_module.sigma_b3

    @property
    def os1(self):
        return self.covar_module.sigma_w1

    @property
    def os2(self):
        return self.covar_module.sigma_w2

    @property
    def os3(self):
        return self.covar_module.sigma_w3

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
                    b1=self.ls1.item(),
                    w1=self.os1.item(),
                    b2=self.ls2.item(),
                    w2=self.os2.item(),
                    noise=self.noise.item(),
                )

        return self
