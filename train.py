import math
import torch.nn as nn
import torch
from collections import OrderedDict
import tensorflow as tf
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import square_attack.utils as utils
from utils import carlini_loss, clip_grad_norms, init_adv_agents, plot_grad_flow, save_agents_param
from target_model import Net
from env import adv_env
from reinforce_baseline import ExponentialBaseline
from options import get_options
from torchvision import datasets, transforms
import os
from itertools import product
from square_attack import models
from square_attack.attack import square_attack_linf as square_attack
import json


def train(opts):
  torch.manual_seed(opts.seed)
  if not os.path.exists(opts.save_dir) and not (opts.eval_fsgm):
    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
        json.dump(vars(opts), f, indent=True)
  pretrained_model = "./target_model_param/lenet_mnist_model.pth"
  batch_size = opts.batch_size
  device = opts.device
  agents = init_adv_agents(opts)
  mnist_dataset = datasets.MNIST(
      opts.output_dir,
      train=False,
      download=True,
      transform=
        transforms.Compose([
          transforms.ToTensor(),
  ]))
  train_dataset, val_dataset, test_dataset = random_split(mnist_dataset, (7000, 1000, 2000))
  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True
  )
  val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True
  )
  test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
  )

  test_im, test_labels = next(iter(test_loader))

  print(test_im.shape, test_labels.shape)

  if opts.dataset == 'mnist':
    num_classes = 10

  if opts.baseline == 'square_attack':
    if opts.dataset == 'mnist':
      sq_model = models.ModelTF('clp_mnist', batch_size, 0.5)
    elif opts.dataset == 'cifar10':
      sq_model = models.ModelTF('clp_cifar10', batch_size, 0.5)
  else:
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
    SCOREFILE = os.path.expanduser(f"./train_rewards_{opts.model}_{opts.epsilon}_{opts.num_timesteps}_{opts.gamma}.csv")
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
      eval_r, eval_loss, eval_acc, *_ = eval(agents, target_model, val_loader, opts.num_timesteps, device, opts)
      with open(SCOREFILE, "a") as f:
        f.write(f'{",".join(map(str, params + (eval_r.mean().item(), eval_loss.mean().item(), eval_acc.mean().item())))}\n')

  elif opts.baseline == 'square_attack':
    print(test_im.shape)
    logits_clean = sq_model.predict(test_im)
    # print(logits_clean)
    
    # test_labels = np.reshape(test_labels,(100,))
    # print(test_labels)
    corr_classified = logits_clean.argmax(1) == test_labels.numpy()

    # corr_classified = torch.from_numpy(corr_classified)
    # print(test_labels)
    # print(corr_classified)
    y_target = utils.random_classes_except_current(test_labels, num_classes) if opts.targetted else test_labels
    test_labels_onehot = utils.dense_to_onehot(y_target, num_classes)

    #test_labels_onehot = torch.from_numpy(test_labels_onehot)
    print(test_im.dtype)
    n_queries, x_adv = square_attack(sq_model, test_im.numpy(), test_labels_onehot, corr_classified, opts.epsilon, opts.n_epochs,
                                     0.05, False, 'cross_entropy')

  elif not (opts.eval_only or opts.eval_fsgm or opts.eval_plots):
    r, acc = train_epoch(agents, target_model, train_loader, opts)

    plt.figure()
    ax1, = plt.plot(np.array(r))
    plt.xlabel("Batch")
    plt.ylabel("Mean Reward")
    plt.savefig(opts.log_dir + "/reward_plot.png")
    torch.save(torch.tensor(r), opts.log_dir + "/reward_logs.pt")

    plt.figure(2)
    ax2, = plt.plot(np.array(acc))
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.savefig(opts.log_dir + "/acc_plot.png")
    torch.save(torch.tensor(acc), opts.log_dir + "/attack_acc_logs.pt")
    save_agents_param(agents, opts)

  

  elif opts.eval_only:
    agents = init_adv_agents(opts, opts.load_paths)
    r, loss, acc, avg_queries, adv_images, orig_images = eval(agents, target_model, test_loader, opts.num_timesteps, device, opts)
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
  elif opts.eval_fsgm:
    target_model.eval()
    attack_accuracy = 0
    assert batch_size == 1  # Needed for FGSM attack to work
    for j, (orig_data, target) in enumerate(test_loader):
      # Send the data and label to the device
      orig_data, target = orig_data.to(device), target.to(device)
      data = orig_data.clone()
      output = target_model(data)
      init_pred = output.argmax(1)
      for i in range(opts.num_timesteps):
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = target_model(data)

        # Calculate the loss
        if not opts.targetted:
          T = target
        else:
          # Randomly sample a target other than true class
          T = torch.ones(opts.batch_size, 10) / 9
          T = T.scatter(1, target, 0)
          T = T.multinomial()
          print(T.shape)
        loss = carlini_loss(output, T)

        # Zero all existing gradients
        target_model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, opts.alpha, data_grad)
        clip_mask = torch.ones((data.size(2), data.size(3)), device=opts.device)
        data = torch.clamp(perturbed_data, min=(orig_data - clip_mask * opts.epsilon), max=(orig_data + clip_mask * opts.epsilon))
        data = torch.clip(data, min=0, max=1)
      with torch.no_grad():
        output2 = target_model(data)
        attack_accuracy += (init_pred != output2.argmax(1)).float().item()
      if j % 200 == 0 and j != 0:
        print(attack_accuracy / j)
  elif opts.eval_plots:
    kernel_sizes = [4, 9]
    eps = [0.1, 0.15, 0.2, 0.25, 0.3]

    for k in kernel_sizes:
      attack_acc = []
      for epsilon in eps:
        opts.k = k
        opts.epsilon = epsilon
        opts.alpha = epsilon

        kernel_size = int(k ** 0.5)
        padded_size = int(math.ceil(((784 ** 0.5) / kernel_size)) * kernel_size)
        num_timesteps = int((padded_size ** 2) / k)
        opts.num_timesteps = num_timesteps
        dir = f"/content/drive/MyDrive/AdvRL/outputs/{opts.model}_k={opts.k}_eps={opts.epsilon}_alpha={opts.alpha}_{int(num_timesteps / 2)}"
        list_of_files = sorted(
            os.listdir(dir), key=lambda s: int(s[8:12] + s[13:])
        )
        agents = init_adv_agents(opts, [dir + f"/{list_of_files[-1]}/agent_0.pt"])

        r, loss, acc, avg_queries, adv_images, orig_images = eval(agents, target_model, test_loader, opts.num_timesteps, device, opts)
        print(f"k={k} eps={epsilon}: Avg queries={avg_queries.mean().item()} Attack Accuracy={acc.mean().item()}")

        attack_acc.append(acc.mean().item())



def fgsm_attack(image, alpha, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + alpha * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image.detach()

def train_batch(agents, target_model, train_loader, optimizers, baseline, opts):
  loss_fun = carlini_loss
  rewards = []
  acc = []
  for i, (x, y) in enumerate(tqdm(train_loader)):
    x = x.to(torch.device(opts.device)).squeeze(1)
    y = y.to(opts.device)
    if not opts.targetted:
      T = y
    else:
      # Randomly sample a target other than true class
      T = torch.ones(opts.batch_size, 10) / 9
      T = T.scatter(1, y[:, None], 0)
      T = T.multinomial(1).squeeze(1)
    env = adv_env(target_model, opts)
    log_p = env.deploy(agents, x, y, T)
    # print(f"Mean Reward: {-r.mean()}")
    with torch.no_grad():
      out = target_model(env.curr_images.unsqueeze(1))
      out2 = target_model(x.unsqueeze(1))
      attack_accuracy = torch.abs((out2.argmax(1) == T).float().sum() - (out.argmax(1) == T).float().sum()) / x.size(0)
      target_model_loss = loss_fun(out, T)
      target_model_loss2 = loss_fun(out2, T)
    # print(f"Target Model Loss: {target_model_loss.mean()}")
    l2_perturb = 0
    r = -(target_model_loss - target_model_loss2).squeeze(1) + opts.gamma * l2_perturb
    # print(torch.softmax(out, dim=1))
    rewards.append(-r.mean().item())
    acc.append(attack_accuracy.item())
    loss = ((r - baseline.eval(r)) * log_p).mean()
    # loss_log.append(loss.item())
    # average_reward.append(-r.mean().item())
    print(f"Avg Reward : {-r.mean().item()}")
    print(f"Attack Accuracy : {attack_accuracy.item()}")
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
  avg_queries = []
  for i, (x, y) in enumerate(tqdm(train_loader)):
    x = x.to(torch.device(device)).squeeze(1)
    y = y.to(device)
    if not opts.targetted:
      T = y
    else:
      # Randomly sample a target other than true class
      T = torch.ones(opts.batch_size, 10) / 9.
      T = T.scatter(1, y[:, None], 0)
      T = T.multinomial(1).squeeze(1)
    env = adv_env(target_model, opts)
    env.sample_type = "greedy"
    with torch.no_grad():
        env.deploy(agents, x, y, T)
        out = target_model(env.curr_images.unsqueeze(1))
        out1 = target_model(x.unsqueeze(1))
        attack_accuracy = torch.abs((out1.argmax(1) == T).float().sum() - (out.argmax(1) == T).float().sum()) / x.size(0)
    target_model_loss = loss_fun(out, T)
    target_model_loss1 = loss_fun(out1, T)
    #print(f"Target Model Loss: {target_model_loss.mean()}")
    l2_perturb = 0
    r = -(target_model_loss - target_model_loss1) + opts.gamma * l2_perturb
    # print(-r.mean().item())
    # print(torch.softmax(out, dim=1))
    # print(f"Target Model Accuracy: {accuracy}")
    rewards.append(-r.mean().item())
    acc.append(attack_accuracy.item())
    losses.append(target_model_loss.mean().item())
    avg_queries.append((env.steps_needed * (env.steps_needed != time_horizon).float()).sum() / (env.steps_needed != time_horizon).float().sum())
    print(f"Attack Accuracy: {attack_accuracy}")
  return torch.tensor(rewards), torch.tensor(losses), torch.tensor(acc), torch.tensor(avg_queries), env.curr_images, x



if __name__ == "__main__":
  train(get_options())
