from config import get_agent_args
from .agent import Agent


def create_agent(env, device_id=None, agent_args=None):
    if agent_args is None:
        agent_args = get_agent_args(env)
    agent = Agent(**agent_args)
    if device_id is not None:
        agent.to(device_id)
    agent.reset()
    return agent, agent_args
