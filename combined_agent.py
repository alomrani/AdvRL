import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class mal_combined_agent(nn.Module):
  def __init__(self):
    super(mal_combined_agent, self).__init__()
    self.conv1 = nn.Conv2d(2, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    # self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(321, 500)
    self.fc2 = nn.Linear(500, 786)

  def forward(self, images, timestep):
    x = F.relu(F.max_pool2d(self.conv1(images), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = torch.cat((x.view(-1, 320), torch.ones(images.size(0), 1, device=images.device) * timestep), dim=1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

  def sample(self, images, timestep, mask):
    state = torch.cat(
        (
            mask[:, None, :, None].reshape(-1, 1, 28, 28),
            images[:, None, :, :]
        ),
        dim=1
    )
    out = self.forward(state, timestep)
    mu, sigma = out[:, -2], torch.abs(out[:, -1].clone()) + 1e-5
    n = torch.distributions.Normal(mu.detach(), sigma.detach())
    a = n.rsample().detach()
    p = torch.exp(-0.5 *((a - mu) / (sigma))**2) * 1 / (sigma * np.sqrt(2 * np.pi))
    lp = torch.log(p + 1e-5)
    return a, lp, out[:, :-2]