import deepmatcher as dm
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://papers.nips.cc/paper/1358-monotonic-networks.pdf
class MonotonicNetworkLazy(dm.modules.LazyModule):
    def _init(self, hidden_size=10,
              output_size=None, input_size=None):
        # print(input_size)
        self.hidden_size = hidden_size
        self.linear_weight = nn.Parameter(torch.ones(hidden_size * hidden_size, input_size))
        self.linear_bias = nn.Parameter(torch.ones(hidden_size * hidden_size))

    def _forward(self, input):
        groups = F.linear(input, torch.abs(self.linear_weight), self.linear_bias)
        out = torch.min(torch.max(groups.view(-1, self.hidden_size, self.hidden_size), 2)[0], 1)[0]
        out = torch.sigmoid(out)
        return torch.log(torch.stack([out, 1. - out], 1))


class MonotonicNetwork(nn.Module):
    def __init__(self, hidden_size=10, input_size=10):
        super(MonotonicNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.linear_weight = nn.Parameter(torch.ones(hidden_size * hidden_size, input_size))
        self.linear_bias = nn.Parameter(-input_size * 0.5 * torch.ones(hidden_size * hidden_size))

    def forward(self, input):
        groups = F.linear(input, torch.abs(self.linear_weight), self.linear_bias)
        out = torch.min(torch.max(groups.view(-1, self.hidden_size, self.hidden_size), 2)[0], 1)[0]
        return out


