import torch
import torch.nn.functional as F
import numpy as np

class adv_env():

  def __init__(self, target_model, time_horizon, epsilon):
    self.images = None
    self.curr_images = None
    self.reward = None
    entropy = None
    self.epsilon = epsilon
    self.prev_ent = None
    self.target_model = target_model
    self.time_horizon = time_horizon
  def update(self, action):
    action_update = F.one_hot(action[:, 0].long(), num_classes=784).reshape(-1, 28, 28)
    self.curr_images = self.curr_images + action_update * torch.clip(action[:, 1][:, None, None], -self.epsilon, self.epsilon)
    self.curr_images = torch.clip(self.curr_images, min=0., max=1.)
    new_ent = self.calc_entropy(self.curr_images)
    self.reward = (new_ent - self.prev_ent).detach()
    self.prev_ent = new_ent
    return self.reward

  def calc_entropy(self, images):
    images = images.unsqueeze(1)
    initial_pred = self.target_model(images).detach()
    initial_pred = F.softmax(initial_pred, dim=1)
    entropy = (-initial_pred * torch.log2(initial_pred)).sum(1)
    return entropy

  def deploy(self, agents, images):
    self.curr_images = images.clone()
    self.images = images
    self.reward = torch.zeros(images.size(0), device=images.device)
    entropy = self.calc_entropy(images)
    self.prev_ent = entropy
    pixel_agent = agents[0]
    perturb_agent = agents[1]
    R = 0
    log = 0
    batch_size = images.size(0)
    for i in range(self.time_horizon):
      out = pixel_agent(torch.cat((self.curr_images.reshape(-1, 784), torch.ones(batch_size, 1, device=images.device) * (i / self.time_horizon)), dim=1))
      selected_pixels, log_p1 = self.sample_pixels(out)
      pertubation, log_p2 = perturb_agent.sample(torch.cat((self.curr_images.reshape(-1, 784), selected_pixels / 784., torch.ones(batch_size, 1, device=images.device) * (i / self.time_horizon)), dim=1))
      action = torch.cat((selected_pixels, pertubation[:, None]), dim=1)
      r = self.update(action)
      R += r
      log += log_p1.squeeze(1) + log_p2
    # plt.imshow(torch.cat((self.curr_images[0, :, :], self.images[0, :, :]), dim=1))
    # plt.show()

    return -R / entropy, log
  def sample_pixels(self, logits):
    l_p = torch.log_softmax(logits, dim=1)
    selected = l_p.exp().multinomial(1)
    return selected, l_p.gather(1, selected)
