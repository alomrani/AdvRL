import torch.nn as nn
import torch
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from perturb_agent import mal_agent2
from pixel_agent import mal_agent
from target_model import Net
from env import adv_env
from reinforce_baseline import ExponentialBaseline
from options import get_options
from torchvision import datasets, transforms
import os
from itertools import product



def train(opts):
  pretrained_model = "./target_model_param/lenet_mnist_model.pth"
  batch_size = opts.batch_size
  device = opts.device
  agent = mal_agent()
  agent2 = mal_agent2()
  train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
      './',
      train=False,
      download=True,
      transform=
        transforms.Compose([
          transforms.ToTensor(),
        ])),
      batch_size=batch_size,
      shuffle=True
  )

  # Initialize the network
  target_model = Net().to(device)

  # Load the pretrained model
  target_model.load_state_dict(torch.load(pretrained_model, map_location=device))

  # Set the model in evaluation mode. In this case this is for the Dropout layers
  target_model.eval()
  if opts.tune:
    PARAM_GRID = list(product(
            [0.01, 0.001, 0.0001, 0.00001, 0.02, 0.002, 0.0002, 0.003, 0.0003, 0.00003],  # learning_rate
            [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],  # baseline exponential decay
            [0.99, 0.98, 0.97, 0.96, 0.95]  # lr decay
        ))
    # total number of slurm workers detected
    # defaults to 1 if not running under SLURM
    N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

    # this worker's array index. Assumes slurm array job is zero-indexed
    # defaults to zero if not running under SLURM
    this_worker = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    SCOREFILE = os.path.expanduser(f"./val_rewards_{opts.gamma}_{opts.num_cars}.csv")
    max_val = 0.
    best_params = []
    for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):
      torch.manual_seed(opts.seed)
      params = PARAM_GRID[param_ix]
      opts.exp_beta = params[1]
      opts.lr_model = params[0]
      opts.lr_decay = params[2]
      agent = mal_agent()
      agent2 = mal_agent2()
      r, acc = train_epoch(agent, agent2, target_model, train_loader, opts)
      with open(SCOREFILE, "a") as f:
        f.write(f'{",".join(map(str, params + (r.mean(),)))}\n')
      

  elif not opts.eval_only:
    r, acc = train_epoch(agent, agent2, target_model, train_loader, opts)



def train_batch(agent, agent2, target_model, train_loader, optimizers, baseline, time_horizon, device):
  loss_fun = nn.CrossEntropyLoss(reduce=False)
  rewards = []
  acc = []
  for i, (x, y) in enumerate(tqdm(train_loader)):
    x = x.to(torch.device(device)).squeeze(1)
    env = adv_env(target_model, time_horizon)
    r, log_p = env.deploy((agent, agent2), x)
    # print(f"Mean Reward: {-r.mean()}")
    out = target_model(env.curr_images.unsqueeze(1)).detach()
    target_model_loss = loss_fun(out, y)
    print(f"Target Model Loss: {target_model_loss.mean()}")
    r = -target_model_loss
    # print(torch.softmax(out, dim=1))
    accuracy = (out.argmax(1) == y).float().sum() / x.size(0)
    print(f"Target Model Accuracy: {accuracy}")
    rewards.append(-r.mean().item())
    acc.append(accuracy)
    optimizers[0].zero_grad()
    optimizers[1].zero_grad()
    loss = ((r - baseline.eval(r)) * log_p).mean()
    # loss_log.append(loss.item())
    # average_reward.append(-r.mean().item())
    loss.backward()
    optimizers[0].step()
    optimizers[1].step()
  return np.array(rewards), np.array(acc)

def train_epoch(agent, agent2, target_model, train_loader, opts):
  beta = opts.exp_beta
  lr_model = opts.lr_model
  lr_decay= opts.lr_decay
  n_epochs = opts.n_epochs
  time_horizon = opts.num_timesteps
  batch_size = opts.batch_size
  device = opts.device
  optimizer = Adam(agent.parameters(), lr=lr_model)
  optimizer2 = Adam(agent2.parameters(), lr=lr_model)

  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda epoch: lr_decay ** epoch
  )
  lr_scheduler2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer2, lambda epoch: lr_decay ** epoch
  )

  agent.train()
  agent2.train()
  baseline = ExponentialBaseline(beta)
  rewards = []
  accuracies = []
  for epoch in range(n_epochs):
    r, acc = train_batch(agent, agent2, target_model, train_loader, [optimizer, optimizer2], baseline, time_horizon, device)
    lr_scheduler.step()
    lr_scheduler2.step()
    if epoch == 0:
      rewards = torch.tensor(r)
      accuracies = torch.tensor(acc)
    else:
      rewards = torch.cat((rewards, torch.tensor(r)))
      accuracies = torch.cat((accuracies, torch.tensor(acc)))

  return rewards, accuracies

if __name__ == "__main__":
  train(get_options())
