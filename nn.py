import math
import warnings
import os
import tqdm
import torch


class NN(torch.nn.Module):
    def __init__(self, train_x, train_y, widths=[], num_outputs=1, classification=True, std=20.):
        assert classification

        # Modules
        super().__init__()
        self.dim = train_x.size(-1)
        self.widths = list(widths)
        self.num_outputs = num_outputs
        self.classification = classification
        self.std = std
        self.wd = 1 / (self.std * train_x.size(0))

        self.first_layer = torch.nn.Linear(self.dim, self.widths[0])
        torch.nn.init.normal_(self.first_layer.weight)
        torch.nn.init.normal_(self.first_layer.bias)
        layers = []
        for in_width, out_width in zip(self.widths, self.widths[1:] + [self.num_outputs]):
            layer = torch.nn.Linear(in_width, out_width)
            torch.nn.init.normal_(layer.weight)
            torch.nn.init.normal_(layer.bias)
            layers.append(layer)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs):
        res = self.first_layer(inputs / math.sqrt(self.dim))
        for layer, width in zip(self.layers, self.widths):
            res = torch.nn.functional.relu(res)
            res = layer(res / math.sqrt(width))
        return res

    def evaluate(self, train_x, train_y, test_x, test_y, savedir):
        self.eval()
        with torch.no_grad():
            logits = self(test_x.cuda())
            output_dist = torch.distributions.Categorical(logits=logits)
            nll = -output_dist.log_prob(test_y.cuda()).mean(dim=-1).cpu()
            err = torch.ne(logits.argmax(dim=-1).cpu(), test_y.cpu()).float().mean(dim=-1)
        return 0, nll.item(), err.item()

    def optimize(self, train_x, train_y, savedir, lr=0.1, minibatch_size=256, num_iter=20000):
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * num_iter), int(0.75 * num_iter)])
        indices = torch.randperm(train_x.size(0), device=train_x.device)
        iterator = tqdm.tqdm(range(num_iter), desc="Training", disable=os.getenv("PROGBAR"))

        for _ in iterator:
            torch.randperm(train_x.size(0), device=train_x.device, out=indices)
            idx = indices[:minibatch_size]
            x_batch = train_x[idx].cuda()
            y_batch = train_y[idx].cuda()

            optimizer.zero_grad()
            logits = self(x_batch)
            loss = torch.nn.functional.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                err = torch.ne(logits.argmax(dim=-1), y_batch).float().mean()
                iterator.set_postfix(loss=loss.item(), err=err.item())

        return self
