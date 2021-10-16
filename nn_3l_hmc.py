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


class NN3L_HMC(torch.nn.Module):
    def __init__(self, train_x, train_y, width, prev_model=None):
        # Modules
        super().__init__()
        self.width = width
        self.register_buffer("sigma_w1", torch.tensor(1.))
        self.register_buffer("sigma_b1", torch.tensor(1.))
        self.register_buffer("sigma_w2", torch.tensor(1.))
        self.register_buffer("sigma_b2", torch.tensor(1.))
        self.register_buffer("sigma_w3", torch.tensor(1.))
        self.register_buffer("sigma_b3", torch.tensor(1.))
        self.register_buffer("noise", torch.tensor(0.2))

        # GP objects
        self.register_buffer("mean_w1", train_x.new_zeros(train_x.size(-1)))
        self.register_buffer("mean_w2", train_x.new_zeros(self.width))
        self.register_buffer("mean_w3", train_x.new_zeros(self.width))
        self.register_buffer("mean_b", train_x.new_zeros([]))

    @property
    def ls1(self):
        return self.sigma_b1

    @property
    def ls2(self):
        return self.sigma_b2

    @property
    def ls3(self):
        return self.sigma_b3

    @property
    def os1(self):
        return self.sigma_w1

    @property
    def os2(self):
        return self.sigma_w2

    @property
    def os3(self):
        return self.sigma_w3

    def forward(self, inputs, targets=None, subsample=None):
        # Draw samples from the first function
        with pyro.plate("layer_1", self.width, dim=-1):
            w1_dist = pyro.distributions.Normal(self.mean_w1, self.sigma_w1).to_event(1)
            b1_dist = pyro.distributions.Normal(self.mean_b, self.sigma_b1)
            w1 = pyro.sample("w1", w1_dist).transpose(-1, -2)
            b1 = pyro.sample("b1", b1_dist)

        with pyro.plate("layer_2", self.width, dim=-1):
            w2_dist = pyro.distributions.Normal(self.mean_w2, self.sigma_w2).to_event(1)
            b2_dist = pyro.distributions.Normal(self.mean_b, self.sigma_b2)
            w2 = pyro.sample("w2", w2_dist).transpose(-1, -2)
            b2 = pyro.sample("b2", b2_dist)

        with pyro.plate("layer_3", 1, dim=-1):
            w3_dist = pyro.distributions.Normal(self.mean_w3, self.sigma_w3).to_event(1)
            b3_dist = pyro.distributions.Normal(self.mean_b, self.sigma_b3)
            w3 = pyro.sample("w3", w3_dist).transpose(-1, -2)
            b3 = pyro.sample("b3", b3_dist)

        # Observations
        with pyro.plate("output_plate", inputs.size(-2), dim=-1, subsample_size=subsample) as idx:
            if idx is not None:
                inputs = inputs[idx]
                if targets is not None:
                    targets = targets[idx]
            hidden_1 = torch.nn.functional.relu(inputs @ w1 + b1)
            hidden_2 = torch.nn.functional.relu(hidden_1 @ w2 / math.sqrt(self.width) + b2)
            latent_vals = (hidden_2 @ w3) / math.sqrt(self.width) + b3
            output_dist = pyro.distributions.Normal(latent_vals.squeeze(-1), self.noise.sqrt())
            outputs = pyro.sample("outputs", output_dist, obs=targets)

        return output_dist

    def initialize_from_previous_model(self, prev_model, train_x, savedir):
        self.sigma_b1.data.copy_(prev_model.ls1)
        self.sigma_b2.data.copy_(prev_model.ls2)
        self.sigma_b3.data.copy_(prev_model.ls3)
        self.sigma_w1.data.copy_(prev_model.os1)
        self.sigma_w2.data.copy_(prev_model.os2)
        self.sigma_w3.data.copy_(prev_model.os3)
        self.noise.data.copy_(prev_model.noise)

    def predictive(self, inputs, train_x):
        output_dist = self.forward(inputs, targets=None)
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
        w1s = samples["w1"].transpose(-1, -2)
        b1s = samples["b1"]
        w2s = samples["w2"].transpose(-1, -2)
        b2s = samples["b2"]
        num_samples = len(w1s)

        # Compute test RMSE and NLL
        log_probs = torch.zeros(num_samples)
        test_iterator = tqdm.tqdm(range(num_samples), desc="Test Samples", disable=os.getenv("PROGBAR"))
        mean = torch.zeros(train_x.size(0), dtype=train_x.dtype, device=train_x.device)
        eye = torch.eye(train_x.size(0), dtype=train_x.dtype, device=train_x.device)
        for i in test_iterator:
            with torch.no_grad():
                features = torch.nn.functional.relu(train_x @ w1s[i] + b1s[i])
                features = torch.nn.functional.relu(features @ w2s[i] / math.sqrt(self.width) + b2s[i])
                covar = torch.add(
                    features @ features.transpose(-1, -2) * (self.sigma_w3 ** 2 / self.width),
                    self.sigma_b3 ** 2
                ) + eye * self.noise
                log_probs[i] = pyro.distributions.MultivariateNormal(mean, covar).log_prob(train_y)

        return log_probs

    def optimize(
        self, train_x, train_y, savedir, warmup_steps=2000, num_samples=1000,
        target_accept_prob=0.8, max_tree_depth=10, lr=0.1,
        adam_lr=0.01, num_map_iter=100,
    ):
        self.eval()

        # Pre warmup
        w1_param = torch.nn.Parameter(torch.randn(self.width, train_x.size(-1), dtype=train_x.dtype, device=train_x.device))
        w2_param = torch.nn.Parameter(torch.randn(self.width, self.width, dtype=train_x.dtype, device=train_x.device))
        w3_param = torch.nn.Parameter(torch.randn(1, self.width, dtype=train_x.dtype, device=train_x.device))
        b1_param = torch.nn.Parameter(torch.randn(self.width, dtype=train_x.dtype, device=train_x.device))
        b2_param = torch.nn.Parameter(torch.randn(1, dtype=train_x.dtype, device=train_x.device))
        b3_param = torch.nn.Parameter(torch.randn(1, dtype=train_x.dtype, device=train_x.device))

        def guide(inputs, targets=None, subsample=None):
            with pyro.plate("layer_1", self.width, dim=-1):
                w1_dist = pyro.distributions.Delta(w1_param).to_event(1)
                b1_dist = pyro.distributions.Delta(b1_param)
                pyro.sample("w1", w1_dist).transpose(-1, -2)
                pyro.sample("b1", b1_dist)

            with pyro.plate("layer_2", self.width, dim=-1):
                w2_dist = pyro.distributions.Delta(w2_param).to_event(1)
                b2_dist = pyro.distributions.Delta(b2_param)
                pyro.sample("w2", w2_dist).transpose(-1, -2)
                pyro.sample("b2", b2_dist)

            with pyro.plate("layer_3", 1, dim=-1):
                w3_dist = pyro.distributions.Delta(w3_param).to_event(1)
                b3_dist = pyro.distributions.Delta(b3_param)
                pyro.sample("w3", w3_dist).transpose(-1, -2)
                pyro.sample("b3", b3_dist)

        optimizer = torch.optim.Adam(lr=adam_lr, params=[w1_param, w2_param, w3_param, b1_param, b2_param, b3_param])
        elbo = pyro.infer.Trace_ELBO()
        iterator = tqdm.tqdm(range(num_map_iter), desc="Training", disable=os.getenv("PROGBAR"))
        for _ in iterator:
            optimizer.zero_grad()
            loss = elbo.differentiable_loss(self, guide, train_x, train_y, subsample=128)
            loss.backward()
            optimizer.step()
            iterator.set_postfix(loss=loss.item())

        initial_params = {
            "w1": w1_param.detach(),
            "b1": b1_param.detach(),
            "w2": w2_param.detach(),
            "b2": b2_param.detach(),
            "w3": w3_param.detach(),
            "b3": b3_param.detach(),
        }

        # MCMC chain
        if num_samples > 0:
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
        else:
            torch.save(
                dict((key, val.unsqueeze(0)) for key, val in initial_params.items()),
                os.path.join(savedir, "mcmc.pth"),
            )

        return self
