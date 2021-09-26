import math
import torch
import numpy as np
import torch.nn as nn

class mal_agent2(nn.Module):
  def __init__(self):
    super(mal_agent2, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(786, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )

  def forward(self, images):
    return self.net(images)
  
  def sample(self, images):
    out = self.net(images)
    mu, sigma = out[:, 0].float(), torch.abs(out[:, 1]) + 1e-5
    n = torch.distributions.Normal(mu.detach(), sigma.detach())
    a = n.rsample().detach()
    p = torch.exp(-0.5 *((a - mu) / (sigma))**2) * 1 / (sigma * np.sqrt(2 * np.pi))
    lp = torch.log(p + 1e-5)
    return a, lp

