import tqdm
import torch
import gpytorch
from gpytorch.utils.transforms import inv_softplus


class LimitAdditiveRBF_DGP(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, **kwargs):
        super().__init__(has_lengthscale=False)
        self.register_parameter("raw_ls1", torch.nn.Parameter(inv_softplus(torch.tensor(1.))))
        self.register_parameter("raw_os1", torch.nn.Parameter(inv_softplus(torch.tensor(1.))))
        self.register_parameter("raw_ls2", torch.nn.Parameter(inv_softplus(torch.tensor(1.))))
        self.register_parameter("raw_os2", torch.nn.Parameter(inv_softplus(torch.tensor(1.))))

    @property
    def hidden_lengthscale(self):
        return torch.nn.functional.softplus(self.raw_hidden_lengthscale)

    @property
    def hidden_outputscale(self):
        return torch.nn.functional.softplus(self.raw_hidden_outputscale)

    @property
    def ls1(self):
        return torch.nn.functional.softplus(self.raw_ls1)

    @property
    def ls2(self):
        return torch.nn.functional.softplus(self.raw_ls2)

    @property
    def os1(self):
        return torch.nn.functional.softplus(self.raw_os1)

    @property
    def os2(self):
        return torch.nn.functional.softplus(self.raw_os2)

    def forward(self, x1, x2, diag=False, **params):
        ls1 = self.ls1
        ls2 = self.ls2
        os1 = self.os1
        os2 = self.os2
        x1 = x1 / ls1
        x2 = x2 / ls1
        if diag:
            dist_mat = (x1 - x2).pow(2).sum(dim=-1)
        else:
            dist_mat = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(2).sum(dim=-1)
        rbf_kernel = dist_mat.mul(-0.5).exp().mul(os1)
        res = os2 * ls2 / torch.sqrt(2. * (os1 - rbf_kernel) + ls2.pow(2.))
        return res


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, init_noise=None):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = LimitAdditiveRBF_DGP()
        self.likelihood.initialize(noise=0.2 if init_noise is None else init_noise)
        self.init_noise = init_noise

    @property
    def ls1(self):
        return self.covar_module.ls1

    @property
    def ls2(self):
        return self.covar_module.ls2

    @property
    def os1(self):
        return self.covar_module.os1

    @property
    def os2(self):
        return self.covar_module.os2

    @property
    def noise(self):
        return self.likelihood.noise.squeeze()

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
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
        if self.init_noise is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(self.covar_module.parameters(), lr=lr)
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
                    os1=self.os1.item(),
                    os2=self.os2.item(),
                    noise=self.noise.item(),
                )

        return self
