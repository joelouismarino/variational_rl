import copy
import torch


class SynchronousEnv(object):
    """
    Class to perform multiple environment rollouts synchoronously. This class is
    currently used only for Monte Carlo evaluation of actual discounted returns.

    Args:
        env (gym.Env): the base environment
        n_envs (int): number of synchonous copies of the environment
    """
    def __init__(self, env, n_envs):
        self.envs = [copy.deepcopy(env) for _ in range(n_envs)]

    def step(self, actions):
        states, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            state, reward, done, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(torch.tensor(done).to(torch.float32).view(1, 1))
            infos.append(info)
        states = torch.cat(states, axis=0)
        rewards = torch.cat(rewards, axis=0)
        dones = torch.cat(dones, axis=0)
        return states, rewards, dones, infos

    def reset(self):
        return torch.cat([env.reset() for env in self.envs], axis=0)

    def set_state(self, qpos, qvel):
        # sets all environments to the same qpos and qvel
        for env in self.envs:
            env.set_state(qpos=qpos, qvel=qvel)
