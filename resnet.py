import math
import warnings
import os
import tqdm
import torch
from torch import nn
from torchvision.models.resnet import conv3x3


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            x = self.downsample(x)
        # TODO: fix the bug of original Stochatic depth
        residual = self.conv1(residual)
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        x = x + residual
        x = self.relu2(x)

        return x


class DownsampleB(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)


class _ResNet(nn.Module):
    '''Small ResNet for CIFAR & SVHN '''
    def __init__(self, dim, depth=8, width=64, block=BasicBlock, initial_stride=1, num_outputs=10):
        assert (depth - 2) % 6 == 0, 'depth should be one of 6N+2'

        super().__init__()
        self.dim = dim

        n = (depth - 2) // 6
        self.inplanes = width // 4
        self.conv1 = nn.Conv2d(3, width // 4, kernel_size=3, stride=initial_stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width // 4)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, width // 4, n)
        self.layer2 = self._make_layer(block, width // 2, n, stride=2)
        self.layer3 = self._make_layer(block, width, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(width * block.expansion, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.expansion,
                                     stride)

        layers = [block(self.inplanes, planes, stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @property
    def classifier(self):
        return self.fc

    @property
    def num_classes(self):
        return self.fc.weight.size(-2)

    @property
    def num_features(self):
        return self.fc.weight.size(-1)

    def extract_features(self, x):
        print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = x.view(x.size(0), 3, 32, 32)
        return self.fc(self.extract_features(x))


class ResNet(nn.Module):
    '''Small ResNet for CIFAR & SVHN '''
    def __init__(self, train_x, train_y, depth=8, width=64, block=BasicBlock, initial_stride=1, num_outputs=10, classification=True, std=0.2):
        assert classification
        assert (depth - 2) % 6 == 0, 'depth should be one of 6N+2'
        self.std = std
        self.wd = 1 / (self.std * train_x.size(0))

        super().__init__()
        self.model = _ResNet(dim=train_x.size(-1), depth=depth, width=width, block=block, initial_stride=initial_stride, num_outputs=num_outputs)
        print(torch.cuda.device_count())
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
            print(self.model.device_ids, torch.cuda.device_count())

    def forward(self, x):
        return self.model(x)

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

        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=self.wd)
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
