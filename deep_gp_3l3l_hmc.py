import math
import warnings
import os
import tqdm
import torch
import pyro
import gpytorch
from scipy.cluster.vq import kmeans2
from random import choice
from collections import deque


class CholeskyError(RuntimeError):
    pass


def psd_safe_cholesky(A, upper=False, out=None, jitter=None, max_tries=3):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
        :attr:`max_tries` (int, optional):
            Number of attempts (with successively increasing jitter) to make before raising an error.
    """
    try:
        L = torch.cholesky(A, upper=upper, out=out)
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise RuntimeError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(max_tries):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(f"A not p.d., added jitter of {jitter_new:.1e} to the diagonal", RuntimeWarning)
                return L
            except RuntimeError:
                continue
        raise CholeskyError(
            f"Matrix not positive definite after repeatedly adding jitter up to {jitter_new:.1e}. "
            f"Original error on first attempt: {e}"
        )


def rbf_kernel(x1, x2, lengthscale, outputscale):
    x1 = x1 / lengthscale
    x2 = x2 / lengthscale
    dist_mat = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(2).sum(dim=-1)
    kernel = dist_mat.mul(-0.5).exp().mul(outputscale)
    return kernel


def add_rbf_kernel(x1, x2, lengthscale, outputscale):
    x1 = x1 / lengthscale
    x2 = x2 / lengthscale
    dist_mat = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(2)
    kernel = dist_mat.mul(-0.5).exp().mul(outputscale / x1.size(-1)).sum(dim=-1)
    return kernel


class DeepGP3L3L_HMC(torch.nn.Module):
    def __init__(self, train_x, train_y, width, width1=2, prev_model=None):
        # Modules
        super().__init__()
        self.width1 = width1
        self.width2 = width
        self.register_buffer("ls1", torch.tensor(1.))
        self.register_buffer("os1", torch.tensor(1.))
        self.register_buffer("ls2", torch.tensor(1.))
        self.register_buffer("os2", torch.tensor(1.))
        self.register_buffer("ls3", torch.tensor(1.))
        self.register_buffer("os3", torch.tensor(1.))
        self.register_buffer("noise", torch.tensor(0.2))

        # GP objects
        self.register_buffer("mean", train_x.new_zeros(train_x.shape[:-1]))
        self.register_buffer("ones", train_x.new_ones(train_x.shape[:-1]))
        self.register_buffer("eye", torch.eye(train_x.size(-2), dtype=train_x.dtype, device=train_x.device))

    def forward(self, inputs, targets=None, train_x=None):
        if train_x is None:
            train_x = inputs

        # Draw samples from the first function
        hidden_covar1 = rbf_kernel(train_x, train_x, self.ls1, self.os1).add(self.eye.mul(1e-4))
        hidden_chol_covar1 = psd_safe_cholesky(hidden_covar1)
        with pyro.plate("hidden_fns1_plate", self.width1, dim=-1):
            hidden_dist = pyro.distributions.Normal(self.mean, self.ones).to_event(1)
            hidden_fns1 = pyro.sample("hidden_fns1", hidden_dist).transpose(-1, -2)
            unwhitened_hidden_fns1 = hidden_chol_covar1 @ hidden_fns1

        # Draw samples from the second function
        hidden_covar2 = add_rbf_kernel(unwhitened_hidden_fns1, unwhitened_hidden_fns1, self.ls2, self.os2).add(self.eye.mul(1e-4))
        hidden_chol_covar2 = psd_safe_cholesky(hidden_covar2)
        with pyro.plate("hidden_fns2_plate", self.width2, dim=-1):
            hidden_dist = pyro.distributions.Normal(self.mean, self.ones).to_event(1)
            hidden_fns2 = pyro.sample("hidden_fns2", hidden_dist).transpose(-1, -2)
            unwhitened_hidden_fns2 = hidden_chol_covar2 @ hidden_fns2

        final_covar = add_rbf_kernel(unwhitened_hidden_fns2, unwhitened_hidden_fns2, self.ls3, self.os3).add(self.eye.mul(1e-4))
        final_chol_covar = psd_safe_cholesky(final_covar)
        with pyro.plate("output_fns_plate", 1, dim=-1):
            final_dist = pyro.distributions.Normal(self.mean, self.ones).to_event(1)
            final_fns = pyro.sample("final_fns", final_dist).transpose(-1, -2)
            unwhitened_final_fns = final_chol_covar @ final_fns

        if torch.equal(train_x, inputs):
            latent_vals = unwhitened_final_fns.squeeze(-1)
        else:
            hidden_cross_covar1 = rbf_kernel(train_x, inputs, self.ls1, self.os1)
            hidden_interp_term1 = torch.triangular_solve(hidden_cross_covar1, hidden_chol_covar1, upper=False)[0]
            pred_hidden_fns1 = hidden_interp_term1.transpose(-1, -2) @ hidden_fns1

            hidden_cross_covar2 = add_rbf_kernel(unwhitened_hidden_fns1, pred_hidden_fns1, self.ls2, self.os2)
            hidden_interp_term2 = torch.triangular_solve(hidden_cross_covar2, hidden_chol_covar2, upper=False)[0]
            pred_hidden_fns2 = hidden_interp_term2.transpose(-1, -2) @ hidden_fns2

            final_cross_covar = add_rbf_kernel(unwhitened_hidden_fns2, pred_hidden_fns2, self.ls3, self.os3)
            final_interp_term = torch.triangular_solve(final_cross_covar, final_chol_covar, upper=False)[0]
            latent_vals = (final_interp_term.transpose(-1, -2) @ final_fns).squeeze(-1)

        # Observations
        with pyro.plate("output_plate", inputs.size(-2), dim=-1):
            output_dist = pyro.distributions.Normal(latent_vals.squeeze(-1), self.noise.sqrt())
            outputs = pyro.sample("outputs", output_dist, obs=targets)

        return output_dist

    def initialize_from_previous_model(self, prev_model, train_x, savedir):
        self.ls1.data.copy_(prev_model.ls1)
        self.ls2.data.copy_(prev_model.ls2)
        self.ls3.data.copy_(prev_model.ls3)
        self.os1.data.copy_(prev_model.os1)
        self.os2.data.copy_(prev_model.os2)
        self.os3.data.copy_(prev_model.os3)
        self.noise.data.copy_(prev_model.noise)

    def predictive(self, inputs, train_x):
        output_dist = self.forward(inputs, targets=None, train_x=train_x)

        if len(output_dist.batch_shape) > 1:
            num_mixtures = output_dist.batch_shape[-2]
            output_dist = pyro.distributions.MixtureSameFamily(
                mixture_distribution=pyro.distributions.Categorical(inputs.new_ones(num_mixtures).div(num_mixtures)),
                component_distribution=output_dist.to_event(1),
            )
        return output_dist

    def evaluate(self, train_x, train_y, test_x, test_y, savedir):
        samples = torch.load(os.path.join(savedir, "mcmc.pth"))

        # Compute test RMSE and NLL
        num_samples = len(next(iter(samples.values())))
        log_probs = torch.zeros(num_samples)
        diffs = torch.zeros(num_samples, len(test_y))
        nlls = torch.zeros(num_samples, len(test_y))

        test_iterator = tqdm.tqdm(range(num_samples), desc="Test Samples", disable=os.getenv("PROGBAR"))
        for i in test_iterator:
            sample = dict((key, val[i]) for key, val in samples.items())
            with torch.no_grad(), pyro.poutine.condition(data=sample):
                trace = pyro.poutine.trace(self.forward, graph_type="flat").get_trace(train_x, train_y)
                trace.compute_log_prob()

                for _, site in trace.nodes.items():
                    if site["type"] == "sample" and site["log_prob"].dim():
                        log_probs[i] += site["log_prob"].sum(dim=-1).cpu()


            with torch.no_grad(), pyro.poutine.condition(data=sample):
                output_dist = self.predictive(test_x, train_x)
                nlls[i] = -output_dist.log_prob(test_y).cpu()
                diffs[i] = (output_dist.mean - test_y).cpu()

        nmll = -(log_probs.logsumexp(dim=0) - math.log(len(log_probs))).div_(len(train_y))
        nll = -((-nlls).logsumexp(dim=0) - math.log(len(nlls))).mean(dim=-1)
        rmse = diffs.mean(0).pow(2).mean(dim=-1).sqrt()
        return nmll.item(), nll.item(), rmse.item()

    def kernel_fit(self, train_x, train_y, savedir):
        samples = torch.load(os.path.join(savedir, "mcmc.pth"), map_location=train_x.device)
        chol_hidden_covar = psd_safe_cholesky(
            rbf_kernel(train_x, train_x, self.ls1, self.os1).add(self.eye.mul(1e-4))
        )
        hidden_fns1 = chol_hidden_covar @ samples["hidden_fns1"].transpose(-1, -2)
        num_samples = len(hidden_fns1)

        # Compute test RMSE and NLL
        log_probs = torch.zeros(num_samples)
        test_iterator = tqdm.tqdm(range(num_samples), desc="Test Samples", disable=os.getenv("PROGBAR"))
        for i in test_iterator:
            with torch.no_grad():
                chol_hidden_covar = psd_safe_cholesky(
                    add_rbf_kernel(hidden_fns1[i], hidden_fns1[i], self.ls2, self.os2).add(self.eye.mul(1e-4))
                )
                hidden_fns2 = chol_hidden_covar @ samples["hidden_fns2"][i].transpose(-1, -2)
                covar = add_rbf_kernel(
                    hidden_fns2, hidden_fns2, lengthscale=self.ls3, outputscale=self.os3
                ) + self.eye * self.noise
                log_probs[i] = pyro.distributions.MultivariateNormal(self.mean, covar).log_prob(train_y)

        return log_probs

    def optimize(
        self, train_x, train_y, savedir, warmup_steps=500, num_samples=500,
        target_accept_prob=0.8, max_tree_depth=10, lr=0.1,
        adam_lr=0.01, num_map_iter=1000,
    ):
        self.eval()

        # Pre warmup
        hidden_fns1_param = torch.nn.Parameter(torch.randn(self.width1, train_x.size(-2), dtype=train_x.dtype, device=train_x.device))
        hidden_fns2_param = torch.nn.Parameter(torch.randn(self.width2, train_x.size(-2), dtype=train_x.dtype, device=train_x.device))
        final_fns_param = torch.nn.Parameter(torch.randn(1, train_x.size(-2), dtype=train_x.dtype, device=train_x.device))

        def guide(inputs, targets=None, train_x=None):
            with pyro.plate("hidden_fns1_plate", self.width1, dim=-1):
                hidden_dist = pyro.distributions.Delta(hidden_fns1_param).to_event(1)
                hidden_fns = pyro.sample("hidden_fns1", hidden_dist).transpose(-1, -2)
            with pyro.plate("hidden_fns2_plate", self.width2, dim=-1):
                hidden_dist = pyro.distributions.Delta(hidden_fns2_param).to_event(1)
                hidden_fns = pyro.sample("hidden_fns2", hidden_dist).transpose(-1, -2)
            with pyro.plate("output_fns_plate", 1, dim=-1):
                final_dist = pyro.distributions.Delta(final_fns_param).to_event(1)
                pyro.sample("final_fns", final_dist).transpose(-1, -2)

        optimizer = torch.optim.Adam(lr=adam_lr, params=[hidden_fns1_param, hidden_fns2_param, final_fns_param])
        elbo = pyro.infer.Trace_ELBO()
        iterator = tqdm.tqdm(range(num_map_iter), desc="Training", disable=os.getenv("PROGBAR"))
        for _ in iterator:
            optimizer.zero_grad()
            loss = elbo.differentiable_loss(self, guide, train_x, train_y)
            loss.backward()
            optimizer.step()
            iterator.set_postfix(loss=loss.item())

        # MCMC chain
        initial_params = {
            "hidden_fns1": hidden_fns1_param.detach(),
            "hidden_fns2": hidden_fns2_param.detach(),
            "final_fns": final_fns_param.detach(),
        }
        kernel = pyro.infer.NUTS(
            self,
            step_size=(lr / len(train_x)),
            adapt_mass_matrix=True,
            adapt_step_size=True,
            max_plate_nesting=1,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
        )
        mcmc = pyro.infer.MCMC(
            kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            initial_params=initial_params,
            num_chains=1,
            disable_progbar=(os.getenv("PROGBAR")),
        )
        mcmc.run(train_x, train_y)
        torch.save(mcmc.get_samples(), os.path.join(savedir, "mcmc.pth"))
        return self
