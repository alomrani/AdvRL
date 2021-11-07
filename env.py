import torch
from torch.autograd import grad
import torch.nn.functional as F
import numpy as np
from utils import carlini_loss

class adv_env():

  def __init__(self, target_model, opts):
    self.images = None
    self.curr_images = None
    self.mask = torch.zeros(opts.batch_size, int(opts.d / opts.k), device=opts.device)
    self.epsilon = opts.epsilon
    self.target_model = target_model
    self.time_horizon = opts.num_timesteps
    self.device = opts.device
    self.opts = opts
    self.sample_type = "sample"
    self.curr_grad_zero = 0.
    self.curr_grad_pos = 0.
    self.curr_grad_neg = 0.
    self.delta = opts.delta
    self.timestep = 0
  def update(self, selected_pixels, selected_mask, grad_estimate=None):
    self.mask = torch.scatter(self.mask, 1, selected_pixels, 1)
    if grad_estimate is None:
      with torch.no_grad():
        x_right = torch.log_softmax(self.target_model(torch.clip(self.curr_images.unsqueeze(1) + selected_mask * self.delta, min=0., max=1.)), dim=1)
        x_left = torch.log_softmax(self.target_model(torch.clip(self.curr_images.unsqueeze(1) - selected_mask * self.delta, min=0., max=1.)), dim=1)

      loss_right = carlini_loss(x_right, self.targets)
      loss_left = carlini_loss(x_left, self.targets)

      grad_estimate = (loss_right - loss_left) / (2 * self.delta)
    self.curr_images = self.curr_images + selected_mask.squeeze(1) * torch.sign(grad_estimate).unsqueeze(2) * self.epsilon
    self.curr_images = torch.clip(self.curr_images, min=0., max=1.)
    return

  def deploy(self, agents, images, true_targets):
    self.curr_images = images.clone()
    self.targets = true_targets
    self.images = images
    self.d = self.opts.d
    log = 0
    self.device = images.device
    for i in range(self.time_horizon):
      selected_pixels, selected_mask, lp_pixel, grad_est, lp_grad_est = self.call_agents(agents, i)
      self.update(selected_pixels, selected_mask, grad_estimate=grad_est)
      log += lp_pixel + lp_grad_est.squeeze(1)
      self.timestep += 1

    return log

  def call_agents(self, agents, timestep):
    selected_grad = None
    lp_grad = torch.tensor([[0.]])
    if self.opts.model == "combined_mal":
      box_agent = agents[0]
      grad_agent = agents[1]
      out_pixel = box_agent.sample(self.images, self.curr_images, timestep / self.time_horizon, self.mask)
      selected_pixels, selected_mask, lp_pixel = self.get_selected_pixels(out_pixel, self.mask)
      out_grad_est = grad_agent.sample(self.images, self.curr_images, timestep / self.time_horizon, self.mask, selected_pixels)
      lp_grad = torch.log_softmax(out_grad_est, dim=1)
      selected_grad = self.sample(lp_grad.exp())
      lp_grad = lp_grad.gather(1, selected_grad)
    elif self.opts.model == "fda_mal":
      box_agent = agents[0]
      out_pixel = box_agent.sample(self.images, self.curr_images, timestep / self.time_horizon, self.mask)
      selected_pixels, selected_mask, lp_pixel = self.get_selected_pixels(out_pixel, self.mask)
    elif self.opts.model == "rg" or self.opts.model == "sg":
      g_attack = agents[0]
      selected_mask, lp_pixel = g_attack(timestep)
      selected_pixels = torch.ones(self.opts.batch_size, 1, device=self.device, dtype=torch.int64) * timestep
    return selected_pixels, selected_mask, lp_pixel, selected_grad, lp_grad


  def get_selected_pixels(self, logits, mask):
    if self.opts.mask:
      logits[mask.bool()] = -1e6
    if self.opts.model == "combined_mal" or self.opts.model == "fda_mal":
      kernel_size = int(self.opts.k ** 0.5)
      l_p = torch.log_softmax(logits, dim=1)
      selected_block = self.sample(l_p.exp())
      selected_mask = F.unfold(torch.zeros(self.images.shape, device=self.device).reshape(-1, 28, 28).unsqueeze(1), kernel_size, stride=kernel_size).transpose(1, 2)
      selected_mask = selected_mask.scatter(1, selected_block[:, :, None].repeat(1, 1, kernel_size ** 2), 1)
      selected_mask = F.fold(selected_mask.transpose(1, 2), kernel_size=kernel_size, stride=kernel_size, output_size=int(self.d ** 0.5))
      return selected_block, selected_mask, l_p.gather(1, selected_block).squeeze(1)
    else:
      l_p = torch.log_softmax(logits, dim=1)
      selected = self.sample(l_p.exp())
      selected_mask = torch.zeros(self.opts.batch_size, 784, device=self.device).scatter(-1, selected, 1)
      return selected_mask.reshape(-1, 28, 28), l_p.gather(1, selected).sum(-1)

  def sample(self, prob):
    assert (prob == prob).all(), "Probs should not contain any nans"
    if self.sample_type == "sample":
      selected = prob.multinomial(num_samples=1)
    else:
      selected = torch.topk(prob, 1, dim=-1)[1]
    return selected