from .discriminative_agent import DiscriminativeAgent
from .generative_agent import GenerativeAgent

def get_agent(agent_args):
    args = agent_args.copy()
    agent_type = args.pop('agent_type')
    if agent_type == 'discriminative':
        return DiscriminativeAgent(**args)
    elif agent_type == 'generative':
        return GenerativeAgent(**args)
    else:
        raise NotImplementedError
