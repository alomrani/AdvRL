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
    self.opts = opts
    self.sample_type = "sample"
  def update(self, selected_pixels, perturb_val):
    self.mask = (self.mask + selected_pixels.reshape(-1, 784) > 0).long()
    self.curr_images = self.curr_images + selected_pixels * torch.clip(perturb_val[:, None, None], -self.epsilon, self.epsilon)
    self.curr_images = torch.clip(self.curr_images, min=0., max=1.)
    return

  def deploy(self, agents, images):
    self.curr_images = images.clone()
    self.images = images
    log = 0
    self.device = images.device
    for i in range(self.time_horizon):
      selected_pixels, perturb_val, lp_pixel, lp_perturb = self.call_agents(agents, i)
      self.update(selected_pixels, perturb_val)
      log += lp_pixel + lp_perturb
    # plt.imshow(torch.cat((self.curr_images[0, :, :], self.images[0, :, :]), dim=1))
    # plt.show()

    return log

  def call_agents(self, agents, timestep):
    batch_size = self.images.size(0)
    if len(agents) == 2:
      pixel_agent = agents[0]
      perturb_agent = agents[1]
      out = pixel_agent(
        torch.cat(
          (
            self.images.reshape(-1, 784),
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
            self.images.reshape(-1, 784),
            self.curr_images.reshape(-1, 784),
            selected_pixels / 784.,
            torch.ones(batch_size, 1, device=self.device) * (timestep / self.time_horizon))
          ),
          dim=1
      )
    else:
      mal_agent = agents[0]
      perturb_val, lp_perturb, out_pixel = mal_agent.sample(self.images, self.curr_images, timestep / self.time_horizon, self.mask)
      selected_pixels, lp_pixel = self.get_selected_pixels(out_pixel, self.mask)
    return selected_pixels, perturb_val, lp_pixel, lp_perturb


  def get_selected_pixels(self, logits, mask):
    if self.opts.mask:
      logits[mask.bool()] = -1e6
    if self.opts.mode == "one-pixel":
      l_p = torch.log_softmax(logits, dim=1)
      selected = self.sample(l_p.exp())
      one_hot_selected = F.one_hot(selected.long(), num_classes=784).reshape(-1, 28, 28)
      return one_hot_selected, l_p.gather(1, selected).squeeze(1)
    else:
      probs = torch.sigmoid(logits)[:, :, None]
      probs = torch.cat((probs, 1. - probs), dim=-1)
      selected = self.sample(probs.reshape(-1, 2)).reshape(-1, 28, 28)
      return selected, (probs.gather(-1, selected) + 1e-6).log().sum((-1, -2))

  def sample(self, prob):
    if self.sample_type == "sample":
      selected = prob.multinomial(num_samples=1)
    else:
      selected = prob.argmax(-1)
    return selected