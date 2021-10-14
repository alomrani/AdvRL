from perturb_agent import mal_agent2
from pixel_agent import mal_agent
from combined_agent import mal_combined_agent
import torch


def init_adv_agents(opts, agents_param_paths=[]):
  if opts.model == "combined_mal":
    agents = [mal_combined_agent().to(opts.device)]
  else:
    agent = mal_agent().to(opts.device)
    agent2 = mal_agent2().to(opts.device)
    agents = [agent, agent2]

  for i, agent_param_path in enumerate(agents_param_paths):
    agent_param = torch.load(agent_param_path, map_location=opts.device)
    agents[i].load_state_dict(agent_param)

  return agents

def save_agents_param(agents, opts):
  for i, agent in enumerate(agents):
    torch.save(agent.state_dict(), opts.save_dir + f"/agent_{i}.pt")