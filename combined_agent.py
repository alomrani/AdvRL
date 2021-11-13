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
    self.fc1 = nn.Linear(322, 200)
    self.fc2 = nn.Linear(200, int(opts.d / opts.k))

  def forward(self, images, timestep, loss):
    x = F.relu(F.max_pool2d(self.conv1(images), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = torch.cat((x.view(-1, 320), torch.ones(images.size(0), 1, device=images.device) * timestep, loss), dim=1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

  def sample(self, images, adv_images, timestep, mask, targets, target_model, loss):
    state = torch.cat(
        (
            images[:, None, :, :],
            adv_images[:, None, :, :],
        ),
        dim=1
    )
    if loss is None:
      with torch.no_grad():
        loss = carlini_loss(target_model(adv_images.unsqueeze(1)), targets)
    out = self.forward(state, timestep, loss)
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

def carlini_loss(output, targets):
  one_hot_targets = F.one_hot(targets.long(), num_classes=10)
  logit_loss = output.gather(1, targets[:, None])
  output[one_hot_targets.bool()] = -1e8
  logit_loss1, _ = output.max(1)
  loss = logit_loss1[:, None] - logit_loss
  return loss
