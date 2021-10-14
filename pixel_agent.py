import math
import torch.nn as nn
class mal_agent(nn.Module):
  def __init__(self):
    super(mal_agent, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(785 + 784, 2000),
        nn.ReLU(),
        nn.Linear(2000, 800),
        nn.ReLU(),
        nn.Linear(800, 784)
    )

  def forward(self, images):
    return self.net(images)
