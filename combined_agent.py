import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class box_agent(nn.Module):
  def __init__(self, opts):
    super(box_agent, self).__init__()
    self.conv1 = nn.Conv2d(2, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    # self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(321, 200)
    self.fc2 = nn.Linear(200, int(opts.d / opts.k))

  def forward(self, images, timestep):
    x = F.relu(F.max_pool2d(self.conv1(images), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = torch.cat((x.view(-1, 320), torch.ones(images.size(0), 1, device=images.device) * timestep), dim=1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

  def sample(self, images, adv_images, timestep, mask):
    state = torch.cat(
        (
            images[:, None, :, :],
            adv_images[:, None, :, :],
        ),
        dim=1
    )
    out = self.forward(state, timestep)
    return out


class grad_agent(nn.Module):
  def __init__(self, opts):
    super(grad_agent, self).__init__()
    self.conv1 = nn.Conv2d(2, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    # self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(322, 150)
    self.fc2 = nn.Linear(150, 2)
    self.opts = opts

  def forward(self, images, timestep, idx):
    x = F.relu(F.max_pool2d(self.conv1(images), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = torch.cat((x.view(-1, 320), torch.ones(images.size(0), 1, device=images.device) * timestep, idx * self.opts.k / self.opts.num_timesteps), dim=1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

  def sample(self, images, adv_images, timestep, mask, idx):
    state = torch.cat(
        (
            images[:, None, :, :],
            adv_images[:, None, :, :],
        ),
        dim=1
    )
    out = self.forward(state, timestep, idx)
    return out
