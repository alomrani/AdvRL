import torch
from torch.autograd import grad
import torch.nn.functional as F
import numpy as np
from utils import carlini_loss

class adv_env():

  def __init__(self, target_model, opts):
    self.images = None
    self.curr_images = None
    self.mask = torch.zeros(opts.batch_size, 784, device=opts.device)
    self.epsilon = opts.epsilon
    self.target_model = target_model
    self.time_horizon = opts.num_timesteps
    self.device = opts.device
    self.opts = opts
    self.sample_type = "sample"
    self.delta = opts.delta
  def update(self, selected_pixels):
    # print(selected_pixels.sum((1, 2)))
    self.mask = (self.mask + selected_pixels.reshape(-1, 784) > 0).long()
    with torch.no_grad():
      x_right = torch.log_softmax(self.target_model(torch.clip(self.curr_images.unsqueeze(1) + selected_pixels.unsqueeze(1) * self.delta, min=0., max=1.)), dim=1)
      x_left = torch.log_softmax(self.target_model(torch.clip(self.curr_images.unsqueeze(1) - selected_pixels.unsqueeze(1) * self.delta, min=0., max=1.)), dim=1)

    loss_right = carlini_loss(x_right, self.targets)
    loss_left = carlini_loss(x_left, self.targets)

    grad_estimate = (loss_right - loss_left) / (2 * self.delta)
    self.curr_images = self.curr_images + selected_pixels * torch.sign(grad_estimate).unsqueeze(-1) * self.epsilon
    self.curr_images = torch.clip(self.curr_images, min=0., max=1.)
    return

  def deploy(self, agents, images, true_targets):
    self.curr_images = images.clone()
    self.targets = true_targets
    self.images = images
    log = 0
    self.device = images.device
    for i in range(self.time_horizon):
      selected_pixels, lp_pixel = self.call_agents(agents, i)
      self.update(selected_pixels)
      log += lp_pixel
      # num_pix_timestep += selected_pixels.sum((1, 2))[:, None]
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
      # perturb_val, lp_perturb = perturb_agent.sample(
      #   torch.cat(
      #     (
      #       self.images.reshape(-1, 784),
      #       self.curr_images.reshape(-1, 784),
      #       selected_pixels / 784.,
      #       torch.ones(batch_size, 1, device=self.device) * (timestep / self.time_horizon))
      #     ),
      #     dim=1
      # )
    elif self.opts.model == "combined_mal":
      mal_agent = agents[0]
      out_pixel = mal_agent.sample(self.images, self.curr_images, timestep / self.time_horizon, self.mask)
      selected_pixels, lp_pixel = self.get_selected_pixels(out_pixel, self.mask)
    elif self.opts.model == "rg":
      rg_attack = agents[0]
      selected_pixels, lp_pixel = rg_attack(timestep)
    elif self.opts.model == "gaussian_agent":
      mal_agent = agents[0]
      out_pixel = mal_agent.sample(self.images, self.curr_images, timestep / self.time_horizon, self.mask)
      selected_pixels, lp_pixel = self.get_selected_pixels(out_pixel, self.mask)
    return selected_pixels, lp_pixel


  def get_selected_pixels(self, logits, mask):
    if self.opts.mask:
      logits[mask.bool()] = -1e6
    if self.opts.mode == "one-pixel":
      l_p = torch.log_softmax(logits, dim=1)
      selected = self.sample(l_p.exp())
      one_hot_selected = F.one_hot(selected.long(), num_classes=784).reshape(-1, 28, 28)
      return one_hot_selected, l_p.gather(1, selected).squeeze(1)
    elif self.opts.mode == "sigmoid-sample":
      probs = torch.sigmoid(logits)[:, :, None]
      probs = torch.cat((1. - probs, probs), dim=-1)
      selected = self.sample(probs.reshape(-1, 2)).reshape(-1, 28, 28)
      return selected, (probs.gather(-1, selected) + 1e-6).log().sum((-1, -2)) / 784
    else:
      l_p = torch.log_softmax(logits, dim=1)
      selected = self.sample(l_p.exp())
      mask_selected = torch.zeros(self.opts.batch_size, 784, device=self.device).scatter_(-1, selected, 1.)
      return mask_selected.reshape(-1, 28, 28), l_p.gather(1, selected).sum(-1)

  def sample(self, prob):
    if self.sample_type == "sample":
      selected = prob.multinomial(num_samples=self.opts.k)
    else:
      selected = torch.topk(prob, self.opts.k, dim=-1)[1]
    return selected