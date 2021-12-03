import torch
import torch.nn as nn

class RGAttack(nn.Module):
    def __init__(self, d, opts) -> None:
        super(RGAttack, self).__init__()
        self.indices = torch.arange(0, d, device=opts.device)[None, :].repeat(opts.batch_size, 1)
        self.k = opts.k
        self.d = d
        self.batch_size = opts.batch_size
        self.opts = opts
    def forward(self, timestep):
        if timestep == 0 or not self.opts.mask:
            r = torch.randperm(self.d)
            # r = torch.randint(0, self.d, size=self.indices.shape)
            self.indices = self.indices[:, r]

        selected_mask = torch.zeros(self.batch_size, self.d, device=self.indices.device)
        selected_indices = self.indices[:, self.k * timestep : self.k * timestep + self.k]
        selected_mask = selected_mask.scatter_(-1, selected_indices, 1)
        return selected_mask.reshape(-1, int(self.d ** 0.5), int(self.d ** 0.5)).unsqueeze(1), 0