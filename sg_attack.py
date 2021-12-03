import torch
import torch.nn as nn
import math

class SGAttack(nn.Module):
    def __init__(self, d, opts) -> None:
        super(SGAttack, self).__init__()
        self.indices = torch.arange(0, d, device=opts.device).reshape(int(d ** 0.5), int(d ** 0.5))
        self.k = opts.k
        self.d = d
        self.batch_size = opts.batch_size
        self.opts = opts
        self.total = opts.num_timesteps
    def forward(self, timestep):

        selected_mask = torch.zeros(self.d, device=self.indices.device)
        start_row = int((timestep * self.k ** 0.5) // ((self.d ** 0.5))) * int(self.k ** 0.5)
        start_col = int((timestep * self.k ** 0.5) % (self.d ** 0.5))
        selected_indices = self.indices[start_row : start_row + int(self.k ** 0.5), start_col : start_col + int(self.k ** 0.5)].flatten()
        selected_mask = selected_mask.scatter_(-1, selected_indices, 1)
        return selected_mask[None, :].repeat(self.batch_size, 1).reshape(-1, int(self.d ** 0.5), int(self.d ** 0.5)).unsqueeze(1), 0
