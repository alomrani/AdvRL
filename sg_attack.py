import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SGAttack(nn.Module):
    def __init__(self, d, opts) -> None:
        super(SGAttack, self).__init__()
        self.indices = torch.arange(0, d, device=opts.device).reshape(int(d ** 0.5), int(d ** 0.5))
        self.k = opts.k
        self.d = d
        self.h = int(opts.d ** 0.5)
        size_pad = {(32, 9): 2, (32, 4): 0, (28, 4): 0, (28, 9): 1}
        self.kernel_size = int(opts.k ** 0.5)
        self.padding = size_pad[(self.h, opts.k)]
        self.padded_size = self.h + 2 * self.padding
        self.batch_size = opts.batch_size
        self.opts = opts
        self.total = opts.num_timesteps
    def forward(self, timestep):
        selected_block = torch.ones(self.batch_size, 1, device=self.opts.device, dtype=torch.int64) * timestep
        selected_mask = F.unfold(torch.zeros((self.opts.batch_size, 1, self.h, self.h), device=self.opts.device), kernel_size=self.kernel_size, stride=self.kernel_size, padding=self.padding).transpose(1, 2)
        selected_mask = selected_mask.scatter(1, selected_block[:, :, None].repeat(1, 1, self.kernel_size ** 2), 1)
        selected_mask = F.fold(selected_mask.transpose(1, 2), kernel_size=self.kernel_size, stride=self.kernel_size, output_size=self.h, padding=self.padding)

        return selected_mask, 0
