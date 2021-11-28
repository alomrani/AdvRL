from perturb_agent import mal_agent2
from pixel_agent import mal_agent
from combined_agent import box_agent, grad_agent
from rg_attack import RGAttack
from sg_attack import SGAttack
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D
import numpy as np


def init_adv_agents(opts, agents_param_paths=[]):
  if opts.model == "combined_mal":
    agents = [box_agent(opts).to(opts.device), grad_agent(opts).to(opts.device)]
  elif opts.model == "fda_mal":
    agent = box_agent(opts).to(opts.device)
    agents = [agent]
  elif opts.model == "rg":
    agents = [RGAttack(784, opts)]
  elif opts.model == "sg":
    agents = [SGAttack(784, opts)]
  if opts.model not in  ["rg", "sg"]:
    for i, agent_param_path in enumerate(agents_param_paths):
      agent_param = torch.load(agent_param_path, map_location=opts.device)
      agents[i].load_state_dict(agent_param)

  return agents

def save_agents_param(agents, opts):
  for i, agent in enumerate(agents):
    torch.save(agent.state_dict(), opts.save_dir + f"/agent_{i}.pt")

def carlini_loss(output, targets):
  one_hot_targets = F.one_hot(targets.long(), num_classes=10)
  logit_loss = output.gather(1, targets[:, None])
  output[one_hot_targets.bool()] = -1e8
  logit_loss1, _ = output.max(1)
  loss = logit_loss1[:, None] - logit_loss
  return loss

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            max_norm
            if max_norm > 0
            else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    )
    return grad_norms, grad_norms_clipped


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    plt.show()
    plt.savefig("grad.png")
