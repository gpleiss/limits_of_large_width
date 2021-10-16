import os
import math
import tqdm
import torch
import gpytorch
from scipy.cluster.vq import kmeans2


class DeepGPHiddenLayer(gpytorch.models.deep_gps.DeepGPLayer):
    def __init__(self, inducing_points, output_dims, ard=True):
        *batch_shape, num_inducing, input_dims = inducing_points.shape

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=(input_dims if ard else 1))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepGP_SVI(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, train_x, train_y, width, num_inducing=300, additive=False):
        # Construct hidden layer inducing points with kmeans
        with torch.no_grad():
            hidden_layer_inducing_points = torch.randn(num_inducing, train_x.size(-1), dtype=train_x.dtype)
            hidden_layer_inducing_points = torch.tensor(
                kmeans2(train_x.cpu().numpy(), hidden_layer_inducing_points.numpy(), minit='matrix')[0]
            ).to(train_x.device)
            hidden_layer_inducing_points = hidden_layer_inducing_points + torch.randn(
                width, *hidden_layer_inducing_points.shape, dtype=train_x.dtype, device=train_x.device
            ).mul_(0.01)

        # Construct final layer inducing points randomly
        if additive:
            last_layer_inducing_points = torch.randn(width, num_inducing, 1, dtype=train_x.dtype, device=train_x.device)
        else:
            last_layer_inducing_points = torch.randn(num_inducing, width, dtype=train_x.dtype, device=train_x.device)

        # Modules
        super().__init__()
        self.additive = additive
        self.hidden_layer = DeepGPHiddenLayer(hidden_layer_inducing_points, output_dims=width, ard=False)
        self.last_layer = DeepGPHiddenLayer(last_layer_inducing_points, output_dims=None, ard=False)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, inputs):
        with gpytorch.settings.num_likelihood_samples(32):
            hidden_rep1 = self.hidden_layer(inputs)
            if self.additive:
                mean = hidden_rep1.mean.transpose(-1, -2).unsqueeze(-1)
                stdv = hidden_rep1.stddev.transpose(-1, -2)
                hidden_rep1 = gpytorch.distributions.MultitaskMultivariateNormal(mean, gpytorch.lazy.DiagLazyTensor(stdv ** 2))
            output = self.last_layer(hidden_rep1)
            if self.additive:
                output = gpytorch.distributions.MultivariateNormal(
                    output.mean.sum(dim=-2),
                    output.lazy_covariance_matrix.sum(dim=-3),
                )
            return output

    def evaluate(self, train_x, train_y, test_x, test_y, savedir, num_samples=16, batch_size=16):
        with gpytorch.settings.num_likelihood_samples(num_samples), torch.no_grad():
            self.eval()
            mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=len(train_x))
            mll = gpytorch.mlls.DeepApproximateMLL(mll)
            nmll = 0.
            for x_batch, y_batch in zip(train_x.split(batch_size), train_y.split(batch_size)):
                output = self(x_batch)
                nmll += -mll(output, y_batch).mul(len(y_batch))
            nmll /= len(train_y)

        with torch.no_grad():
            nll = 0.
            mse = 0.
            self.eval()

            for x_batch, y_batch in zip(test_x.split(batch_size), test_y.split(batch_size)):
                pred_dist = self.likelihood(self(x_batch))
                pred_dist = torch.distributions.Normal(pred_dist.loc.T, pred_dist.stddev.T)
                mixture_weights = x_batch.new_ones(pred_dist.batch_shape[-1])
                pred_dist = torch.distributions.MixtureSameFamily(
                    mixture_distribution=torch.distributions.Categorical(mixture_weights),
                    component_distribution=pred_dist
                )
                nll += -pred_dist.log_prob(y_batch).sum(dim=-1)
                mse += (pred_dist.mean - y_batch).pow(2).sum(dim=-1)

            nll = nll.div(len(test_y))
            rmse = mse.div(len(test_y)).sqrt()

        return nmll.item(), nll.item(), rmse.item()

    @property
    def ls1(self):
        return self.hidden_layer.covar_module.base_kernel.lengthscale.squeeze()

    @property
    def ls2(self):
        return self.last_layer.covar_module.base_kernel.lengthscale.squeeze()

    @property
    def os1(self):
        return self.hidden_layer.covar_module.outputscale.squeeze()

    @property
    def os2(self):
        return self.last_layer.covar_module.outputscale.squeeze()

    @property
    def noise(self):
        return self.likelihood.noise.squeeze()

    def initialize_from_previous_model(self, prev_model, train_x, savedir):
        self.hidden_layer.covar_module.base_kernel.lengthscale = prev_model.ls1
        self.hidden_layer.covar_module.outputscale = prev_model.os1
        self.last_layer.covar_module.base_kernel.lengthscale = prev_model.ls2
        self.last_layer.covar_module.outputscale = prev_model.os2
        self.likelihood.initialize(noise=prev_model.noise)

    def optimize(self, train_x, train_y, savedir, batch_size=256, num_iter=2000, lr=0.01, milestones=[0.5, 0.75]):
        self.train()
        # self.hidden_layer.covar_module.base_kernel.initialize(lengthscale=4.29)
        # self.hidden_layer.covar_module.initialize(outputscale=2.4)
        # self.last_layer.covar_module.base_kernel.initialize(lengthscale=0.035)
        # self.last_layer.covar_module.initialize(outputscale=1.03)
        # self.likelihood.initialize(noise=0.1)

        optimizer = torch.optim.Adam(self.variational_parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * num_iter), int(0.75 * num_iter)])
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=len(train_x))
        mll = gpytorch.mlls.DeepApproximateMLL(mll)
        iterator = tqdm.tqdm(range(num_iter), desc="ExactGP Training", disable=os.getenv("PROGBAR"))

        with gpytorch.settings.max_cholesky_size(1e6):
            for _ in iterator:
                if batch_size is not None:
                    indices = torch.randperm(len(train_x), device=train_x.device)[:batch_size]
                    sub_train_x = train_x[indices]
                    sub_train_y = train_y[indices]
                else:
                    sub_train_x = train_x
                    sub_train_y = train_y

                # optimizer = full_optimizer if _ > 1000 else var_optimizer
                optimizer.zero_grad()
                output = self(sub_train_x)
                loss = -mll(output, sub_train_y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                iterator.set_postfix(
                    loss=loss.item(),
                    ls1=self.hidden_layer.covar_module.base_kernel.lengthscale.mean().item(),
                    os1=self.hidden_layer.covar_module.outputscale.item(),
                    ls2=self.last_layer.covar_module.base_kernel.lengthscale.item(),
                    os2=self.last_layer.covar_module.outputscale.item(),
                    noise=self.likelihood.noise.item(),
                )

        return self
