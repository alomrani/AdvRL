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
    self.device = None
  def update(self, action):
    self.mask = torch.scatter(self.mask, -1, action[:, 0].unsqueeze(1).long(), 1)
    action_update = F.one_hot(action[:, 0].long(), num_classes=784).reshape(-1, 28, 28)
    self.curr_images = self.curr_images + action_update * torch.clip(action[:, 1][:, None, None], -self.epsilon, self.epsilon)
    self.curr_images = torch.clip(self.curr_images, min=0., max=1.)
    return

  def deploy(self, agents, images):
    self.curr_images = images.clone()
    self.images = images
    log = 0
    self.device = images.device
    total_perturbs = 0
    for i in range(self.time_horizon):
      selected_pixels, perturb_val, lp_pixel, lp_perturb = self.call_agents(agents, i)
      action = torch.cat((selected_pixels, perturb_val[:, None]), dim=1)
      self.update(action)
      log += lp_pixel.squeeze(1) + lp_perturb
      total_perturbs += torch.abs(perturb_val)
    # plt.imshow(torch.cat((self.curr_images[0, :, :], self.images[0, :, :]), dim=1))
    # plt.show()

    return log, total_perturbs

  def call_agents(self, agents, timestep):
    batch_size = self.images.size(0)
    if len(agents) == 2:
      pixel_agent = agents[0]
      perturb_agent = agents[1]
      out = pixel_agent(
        torch.cat(
          (
            self.curr_images.reshape(-1, 784),
            self.mask,
            torch.ones(batch_size, 1, device=self.device) * (timestep / self.time_horizon)),
          ),
          dim=1
      )
      selected_pixels, lp_pixel = self.sample_pixels(out, self.mask)
      perturb_val, lp_perturb = perturb_agent.sample(
        torch.cat(
          (
            self.curr_images.reshape(-1, 784),
            selected_pixels / 784.,
            torch.ones(batch_size, 1, device=self.device) * (timestep / self.time_horizon))
          ),
          dim=1
      )
    else:
      mal_agent = agents[0]
      perturb_val, lp_perturb, out_pixel = mal_agent.sample(self.curr_images, timestep / self.time_horizon, self.mask)
      selected_pixels, lp_pixel = self.sample_pixels(out_pixel, self.mask)

    return selected_pixels, perturb_val, lp_pixel, lp_perturb


  def sample_pixels(self, logits, mask):
    logits[mask.bool()] = -1e6
    l_p = torch.log_softmax(logits, dim=1)
    selected = l_p.exp().multinomial(1)
    return selected, l_p.gather(1, selected)
