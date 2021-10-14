import torch
import torch.nn.functional as F
import numpy as np

class adv_env():

  def __init__(self, target_model, opts):
    self.images = None
    self.curr_images = None
    self.mask = torch.zeros(opts.batch_size, 784, device=opts.device)
    self.epsilon = opts.epsilon
    self.target_model = target_model
    self.time_horizon = opts.num_timesteps
  def update(self, action):
    self.mask = torch.scatter(self.mask, -1, action[:, 0].unsqueeze(1).long(), 1)
    action_update = F.one_hot(action[:, 0].long(), num_classes=784).reshape(-1, 28, 28)
    self.curr_images = self.curr_images + action_update * torch.clip(action[:, 1][:, None, None], -self.epsilon, self.epsilon)
    self.curr_images = torch.clip(self.curr_images, min=0., max=1.)
    return

  # def calc_entropy(self, images):
  #   images = images.unsqueeze(1)
  #   initial_pred = self.target_model(images).detach()
  #   initial_pred = F.softmax(initial_pred, dim=1)
  #   entropy = (-initial_pred * torch.log2(initial_pred)).sum(1)
  #   return entropy

  def deploy(self, agents, images):
    self.curr_images = images.clone()
    self.images = images
    pixel_agent = agents[0]
    perturb_agent = agents[1]
    log = 0
    batch_size = images.size(0)
    for i in range(self.time_horizon):
      out = pixel_agent(torch.cat((self.curr_images.reshape(-1, 784), self.mask, torch.ones(batch_size, 1, device=images.device) * (i / self.time_horizon)), dim=1))
      selected_pixels, log_p1 = self.sample_pixels(out, self.mask)
      pertubation, log_p2 = perturb_agent.sample(torch.cat((self.curr_images.reshape(-1, 784), selected_pixels / 784., torch.ones(batch_size, 1, device=images.device) * (i / self.time_horizon)), dim=1))
      action = torch.cat((selected_pixels, pertubation[:, None]), dim=1)
      self.update(action)
      log += log_p1.squeeze(1) + log_p2
    # plt.imshow(torch.cat((self.curr_images[0, :, :], self.images[0, :, :]), dim=1))
    # plt.show()

    return log
  def sample_pixels(self, logits, mask):
    logits[mask.bool()] = -1e6
    l_p = torch.log_softmax(logits, dim=1)
    selected = l_p.exp().multinomial(1)
    return selected, l_p.gather(1, selected)
