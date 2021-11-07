import torch.nn as nn
import torch
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import carlini_loss, clip_grad_norms, init_adv_agents, plot_grad_flow, save_agents_param
from target_model import Net
from env import adv_env
from reinforce_baseline import ExponentialBaseline
from options import get_options
from torchvision import datasets, transforms
import os
from itertools import product
import json


def train(opts):
  torch.manual_seed(opts.seed)
  if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
        json.dump(vars(opts), f, indent=True)
  pretrained_model = "./target_model_param/lenet_mnist_model.pth"
  batch_size = opts.batch_size
  device = opts.device
  agents = init_adv_agents(opts)
  train_loader = DataLoader(
    datasets.MNIST(
      opts.output_dir,
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
            [1.0, 0.99, 0.98, 0.97, 0.96, 0.95]  # lr decay
        ))
    # total number of slurm workers detected
    # defaults to 1 if not running under SLURM
    N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

    # this worker's array index. Assumes slurm array job is zero-indexed
    # defaults to zero if not running under SLURM
    this_worker = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    SCOREFILE = os.path.expanduser(f"./train_rewards_{opts.mode}_{opts.epsilon}_{opts.num_timesteps}_{opts.gamma}.csv")
    max_val = 0.
    best_params = []
    for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):
      torch.manual_seed(opts.seed)
      params = PARAM_GRID[param_ix]
      opts.exp_beta = params[1]
      opts.lr_model = params[0]
      opts.lr_decay = params[2]
      agents = init_adv_agents(opts)
      r, acc = train_epoch(agents, target_model, train_loader, opts)
      eval_r, eval_loss, eval_acc, *_ = eval(agents, target_model, train_loader, opts.num_timesteps, device, opts)
      with open(SCOREFILE, "a") as f:
        f.write(f'{",".join(map(str, params + (eval_r.mean().item(), eval_loss.mean().item(), eval_acc.mean().item())))}\n')


  elif not opts.eval_only:
    r, acc = train_epoch(agents, target_model, train_loader, opts)
    plt.figure()
    ax1, = plt.plot(np.array(r))
    plt.xlabel("Batch")
    plt.ylabel("Mean Reward")
    plt.savefig(opts.save_dir + "/reward_plot.png")
    plt.figure(2)
    ax2, = plt.plot(np.array(acc))
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.savefig(opts.save_dir + "/acc_plot.png")
    save_agents_param(agents, opts)
  elif opts.eval_only:
    agents = init_adv_agents(opts, opts.load_paths)
    r, loss, acc, adv_images, orig_images = eval(agents, target_model, train_loader, opts.num_timesteps, device, opts)
    print(r.mean().item())
    print(acc.mean().item())
    plt.imsave(opts.save_dir + '/adv_image.png', np.array(adv_images[-1]), cmap='gray')
    plt.imsave(opts.save_dir + '/orig_image.png', np.array(orig_images[-1]), cmap='gray')
    plt.imsave(opts.save_dir + '/adv_image1.png', np.array(adv_images[-2]), cmap='gray')
    plt.imsave(opts.save_dir + '/orig_image1.png', np.array(orig_images[-2]), cmap='gray')
    plt.imsave(opts.save_dir + '/adv_image2.png', np.array(adv_images[0]), cmap='gray')
    plt.imsave(opts.save_dir + '/orig_image2.png', np.array(orig_images[0]), cmap='gray')
    plt.imsave(opts.save_dir + '/adv_image3.png', np.array(adv_images[50]), cmap='gray')
    plt.imsave(opts.save_dir + '/orig_image3.png', np.array(orig_images[50]), cmap='gray')



def train_batch(agents, target_model, train_loader, optimizers, baseline, opts):
  loss_fun = carlini_loss
  rewards = []
  acc = []
  for i, (x, y) in enumerate(tqdm(train_loader)):
    x = x.to(torch.device(opts.device)).squeeze(1)
    y = y.to(opts.device)
    env = adv_env(target_model, opts)
    log_p = env.deploy(agents, x, y)
    # print(f"Mean Reward: {-r.mean()}")
    with torch.no_grad():
      out = target_model(env.curr_images.unsqueeze(1))
      out2 = target_model(env.images.unsqueeze(1))
      attack_accuracy = torch.abs((out2.argmax(1) == y).float().sum() - (out.argmax(1) == y).float().sum()) / x.size(0)
      target_model_loss = loss_fun(out, y)
      target_model_loss2 = loss_fun(out2, y)
    # print(f"Target Model Loss: {target_model_loss.mean()}")
    l2_perturb = 0
    r = -(target_model_loss - target_model_loss2).squeeze(1) + opts.gamma * l2_perturb
    # print(torch.softmax(out, dim=1))
    # print(f"Target Model Accuracy: {accuracy}")
    rewards.append(-r.mean().item())
    acc.append(attack_accuracy.item())
    loss = ((r - baseline.eval(r)) * log_p).mean()
    # loss_log.append(loss.item())
    # average_reward.append(-r.mean().item())
    for optimizer in optimizers:
      optimizer.zero_grad()
    loss.backward()
    # grad_norms = clip_grad_norms(optimizers[0].param_groups, 1.0)
    # grad_norms, grad_norms_clipped = grad_norms
    # print("grad_norm: {}, clipped: {}".format(grad_norms[0], grad_norms_clipped[0]))
    for optimizer in optimizers:
      optimizer.step()
  print(f"Average Reward: {np.array(rewards).mean()}")
  print(f"Attack Accuracy: {np.array(acc).mean()}")
  return np.array(rewards), np.array(acc)

def train_epoch(agents, target_model, train_loader, opts):
  beta = opts.exp_beta
  lr_model = opts.lr_model
  lr_decay= opts.lr_decay
  n_epochs = opts.n_epochs
  time_horizon = opts.num_timesteps
  batch_size = opts.batch_size
  device = opts.device
  optimizers = [Adam(agent.parameters(), lr=lr_model) for agent in agents]

  lr_schedulers = [LambdaLR(optimizer, lambda epoch: lr_decay ** epoch) for optimizer in optimizers]

  for agent in agents:
    agent.train()
  baseline = ExponentialBaseline(beta)
  rewards = []
  accuracies = []
  for epoch in range(n_epochs):
    r, acc = train_batch(agents, target_model, train_loader, optimizers, baseline, opts)
    for lr_scheduler in lr_schedulers:
      lr_scheduler.step()
    if epoch == 0:
      rewards = torch.tensor(r)
      accuracies = torch.tensor(acc)
    else:
      rewards = torch.cat((rewards, torch.tensor(r)))
      accuracies = torch.cat((accuracies, torch.tensor(acc)))

  return rewards, accuracies


def eval(agents, target_model, train_loader, time_horizon, device, opts):
  loss_fun = carlini_loss
  rewards = []
  acc = []
  losses = []
  for i, (x, y) in enumerate(tqdm(train_loader)):
    x = x.to(torch.device(device)).squeeze(1)
    y = y.to(device)
    env = adv_env(target_model, opts)
    env.sample_type = "greedy"
    with torch.no_grad():
        env.deploy(agents, x, y)
        out = target_model(env.curr_images.unsqueeze(1))
        out1 = target_model(env.images.unsqueeze(1))
        attack_accuracy = torch.abs((out1.argmax(1) == y).float().sum() - (out.argmax(1) == y).float().sum()) / x.size(0)
    target_model_loss = loss_fun(out, y)
    target_model_loss1 = loss_fun(out1, y)
    #print(f"Target Model Loss: {target_model_loss.mean()}")
    l2_perturb = 0
    r = -(target_model_loss - target_model_loss1) + opts.gamma * l2_perturb
    # print(torch.softmax(out, dim=1))
    # print(f"Target Model Accuracy: {accuracy}")
    rewards.append(-r.mean().item())
    acc.append(attack_accuracy.item())
    losses.append(target_model_loss.mean().item())
    print(f"Attack Accuracy: {attack_accuracy}")
  return torch.tensor(rewards), torch.tensor(losses), torch.tensor(acc), env.curr_images[:, 4:-4, 4:-4], env.images[:, 4:-4, 4:-4]



if __name__ == "__main__":
  train(get_options())
