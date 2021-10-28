import math
import torch.nn as nn
import torch
import torch.nn.functional as F

class mal_agent(nn.Module):
  def __init__(self):
    super(mal_agent, self).__init__()
    self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    # self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(321, 500)
    self.fc2 = nn.Linear(500, 784)

  def forward(self, images, timestep):
    x = F.relu(F.max_pool2d(self.conv1(images), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = torch.cat((x.view(-1, 320), torch.ones(images.size(0), 1, device=images.device) * timestep), dim=1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
