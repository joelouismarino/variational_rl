import gym
from .drone_model import DroneModel


class DroneEnv(gym.Env):
    """
    Differentiable drone environment, based on model by Guanya Shi.

    Args:
        batch_size (int): the number of parallel environments to use
    """
    def __init__(self, batch_size=1):
        self.model = DroneModel(batch_size)

    def step(self, action):
        """
        Step the model forward.
        """
        next_state, reward, done = self.model.step(action)
        return next_state, reward, done, {}

    def set_state(self, state):
        """
        Set the state of the model.
        """
        self.model.set_state(state)

    def reset(self):
        """
        Reset the state of the environment.
        """
        self.model.reset()
