import copy
import torch
import numpy as np
from torch import optim


def collect_optimized_episode(env, agent, random=False, eval=False,
                              semi_am=True, ngs=50):
    """
    Collects an episode of experience using the model and environment. The
    policy distribution is optimized using gradient descent at each step.

    Args:
        env (gym.env): the environment
        agent (Agent): the agent
        random (bool): whether to use random actions
        eval (bool): whether to evaluate the agent
        semi_am (bool): whether to first use direct inference
        ngs (int): number of gradient steps to perform

    Returns episode (dict), n_steps (int), and env_states (dict).
    """
    agent.reset(); agent.eval()
    state = env.reset()
    reward = 0.
    done = False
    n_steps = 0
    env_states = {'qpos': [], 'qvel': []}

    optimized_actions = []
    gaps = []

    while not done:
        if n_steps > 1000:
            break
        if 'sim' in dir(env.unwrapped):
            env_states['qpos'].append(copy.deepcopy(env.sim.data.qpos))
            env_states['qvel'].append(copy.deepcopy(env.sim.data.qvel))

        action = env.action_space.sample() if random else None
        agent.act(state, reward, done, action, eval=eval)

        ## SEMI - AMORTIZATION #################################################
        state = state.to(agent.device)
        actions = agent.approx_post.sample(agent.n_action_samples)
        obj = agent.estimate_objective(state, actions)
        direct_obj = obj.view(agent.n_action_samples, -1, 1).mean(dim=0).detach()

        agent.n_action_samples = 100
        grad_obj = []
        dist_params = {k: v.data.requires_grad_() for k, v in agent.approx_post.get_dist_params().items()}
        agent.approx_post.reset(dist_params=dist_params)
        dist_param_list = [param for _, param in dist_params.items()]
        optimizer = optim.Adam(dist_param_list, lr=5e-3)
        optimizer.zero_grad()
        # initial estimate
        agent.approx_post._sample = None
        actions = agent.approx_post.sample(agent.n_action_samples)
        obj = agent.estimate_objective(state, actions)
        obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
        grad_obj.append(-obj.detach())

        for it_inf in range(ngs):
            obj.sum().backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            # clear the sample to force resampling
            agent.approx_post._sample = None
            actions = agent.approx_post.sample(agent.n_action_samples)
            obj = agent.estimate_objective(state, actions)
            obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
            grad_obj.append(-obj.detach())
        # gradient_obj = np.array([obj.numpy() for obj in grad_obj]).reshape(-1)
        gaps.append((grad_obj[-1] - grad_obj[0]).cpu().numpy().item())

        # sample from the optimized distribution
        action = agent.approx_post.sample(n_samples=1, argmax=eval)
        action = action.tanh() if agent.postprocess_action else action
        action = action.detach().cpu().numpy()
        optimized_actions.append(action)

        ########################################################################

        # step the environment with the optimized action
        state, reward, done, _ = env.step(action)
        n_steps += 1

    print('     Average Improvement: ' + str(np.mean(gaps)))
    if 'sim' in dir(env.unwrapped):
        env_states['qpos'].append(copy.deepcopy(env.sim.data.qpos))
        env_states['qvel'].append(copy.deepcopy(env.sim.data.qvel))
        env_states['qpos'] = np.stack(env_states['qpos'])
        env_states['qvel'] = np.stack(env_states['qvel'])
    agent.act(state, reward, done)

    episode = agent.collector.get_episode()

    if not random:
        # replace the collector's actions with optimized actions
        final_action = np.zeros(optimized_actions[-1].shape)
        optimized_actions.append(final_action)
        optimized_actions = np.concatenate(optimized_actions, axis=0)
        optimized_actions = torch.from_numpy(optimized_actions).float()
        episode['action'] = optimized_actions

    return episode, n_steps, env_states
