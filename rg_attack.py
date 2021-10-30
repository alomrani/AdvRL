import torch
import torch.nn as nn

class RGAttack(nn.Module):
    def __init__(self, d, opts) -> None:
        super(RGAttack, self).__init__()
        self.indices = torch.arange(0, d, device=opts.device)[None, :].repeat(opts.batch_size, 1)
        self.k = opts.k
        self.d = d
        self.batch_size = opts.batch_size
    def forward(self, timestep):
        if timestep == 0:
            r = torch.randperm(self.d)
            self.indices = self.indices[:, r]
        selected_mask = torch.zeros(self.batch_size, self.d, device=self.indices.device)
        selected_mask = selected_mask.scatter_(-1, self.indices[:, self.k * timestep : self.k * timestep + self.k], 1)
        return selected_mask.reshape(-1, 28, 28), 0