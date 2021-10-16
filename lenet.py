import math
import warnings
import os
import tqdm
import torch
from torch import nn
from torchvision.models.resnet import conv3x3


class LeNet(nn.Module):
    def __init__(self, train_x, train_y, conv_width=16, width=64, num_outputs=10, classification=True, std=20.):
        assert classification
        self.std = std
        self.wd = 1 / (self.std * train_x.size(0))

        super().__init__()
        self.dim = train_x.size(-1)
        self.conv_width = conv_width
        self.width = width
        self.num_outputs = num_outputs

        self.conv1 = torch.nn.Conv2d(3, conv_width, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(conv_width, conv_width, kernel_size=5)
        self.lin3 = torch.nn.Linear(conv_width * 5 * 5, width)
        self.lin4 = torch.nn.Linear(width, width)
        self.lin5 = torch.nn.Linear(width, self.num_outputs)

    def forward(self, x):
        res = x.view(x.size(0), 3, 32, 32)
        res = torch.nn.functional.relu(self.conv1(res / math.sqrt(self.dim)))
        res = torch.nn.functional.avg_pool2d(res, 2, 2)
        res = torch.nn.functional.relu(self.conv2(res / math.sqrt(self.conv_width)))
        res = torch.nn.functional.avg_pool2d(res, 2, 2)
        res = res.view(res.size(0), -1)
        res = torch.nn.functional.relu(self.lin3(res / math.sqrt(self.conv_width * 5 * 5)))
        res = torch.nn.functional.relu(self.lin4(res / math.sqrt(self.width)))
        res = self.lin5(res / math.sqrt(self.width))
        return res

    def evaluate(self, train_x, train_y, test_x, test_y, savedir):
        self.eval()
        with torch.no_grad():
            nlls = []
            errs = []
            for x_batch, y_batch in zip(test_x.split(128), test_y.split(128)):
                logits = self(x_batch.cuda())
                output_dist = torch.distributions.Categorical(logits=logits)
                nll = -output_dist.log_prob(y_batch.cuda()).sum(dim=-1).cpu()
                err = torch.ne(logits.argmax(dim=-1).cpu(), y_batch).float().sum(dim=-1)
                nlls.append(nll.item() / len(test_y))
                errs.append(err.item() / len(test_y))

        return 0, sum(nlls), sum(errs)

    def optimize(self, train_x, train_y, savedir, lr=0.1, minibatch_size=256, num_iter=40000):
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
