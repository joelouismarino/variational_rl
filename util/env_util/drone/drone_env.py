import gym
from .drone_model import DroneModel


class Drone(gym.Env):
    """
    Differentiable drone environment, based on model by Guanya Shi.

    Args:
        batch_size (int): the number of parallel environments to use
    """
    def __init__(self, batch_size):
        self.model = DroneModel(batch_size)

    def step(self, action):
        next_state = self.model.step(action)

        return next_state, reward, done, {}

    def reset(self):
        self.model.reset()
