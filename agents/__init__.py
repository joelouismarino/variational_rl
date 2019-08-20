import torch
from config import get_agent_args
from .baseline_agent import BaselineAgent
from .model_based_agent import ModelBasedAgent
from .discriminative_agent import DiscriminativeAgent
from .generative_agent import GenerativeAgent


def create_agent(env, ckpt_path=None, device_id=None):
    agent_args = get_agent_args(env)
    args = agent_args.copy()
    agent_type = args.pop('agent_type')

    if agent_type == 'baseline':
        agent = BaselineAgent(**args)
    elif agent_type == 'model_based':
        agent = ModelBasedAgent(**args)
    elif agent_type == 'discriminative':
        agent = DiscriminativeAgent(**args)
    elif agent_type == 'generative':
        agent = GenerativeAgent(**args)
    else:
        raise NotImplementedError

    if ckpt_path is not None:
        print('Loading checkpoint from ' + ckpt_path)
        state_dict = torch.load(ckpt_path)
        agent.load(state_dict)

    if device_id is not None:
        agent.to(device_id)
    agent.reset()

    return agent, agent_args
