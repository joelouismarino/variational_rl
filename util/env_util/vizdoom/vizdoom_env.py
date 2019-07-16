# inspired by nsavinov/gym-vizdoom and ppaquette/gym-doom
import numpy as np
import gym.spaces as spaces
from gym import Env
from gym.envs.classic_control import rendering
from vizdoom import DoomGame
import itertools as it
from .reward_ranges import get_reward_range, reward_ranges


class VizDoomEnv(Env):
    '''
    Wrapper for vizdoom to use as an OpenAI gym environment.
    '''
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, cfg_name, repeat=4):
        super(VizDoomEnv, self).__init__()
        self.game = DoomGame()
        self.game.load_config('./envs/vizdoom/cfgs/' + cfg_name + '.cfg')
        self._viewer = None
        self.repeat = repeat # originally is set to 1. Under pytorch implementation that came with vizdoom it's 12.
        # TODO In future, need to update action to handle (continuous) DELTA buttons using gym's Box space
        n_actions = self.game.get_available_buttons_size()
        self.action_combinations = [list(a) for a in it.product([0, 1], repeat=n_actions)]
        n_combinations = len(self.action_combinations)
        # self.action_space = spaces.MultiDiscrete([2] * n_combinations) # Joe's implementation
        self.action_space = spaces.Discrete(n_combinations)
        self.action_space.n = n_combinations
        self.action_space.dtype = 'uint8'
        output_shape = (self.game.get_screen_height(), self.game.get_screen_width(), self.game.get_screen_channels())
        self.observation_space = spaces.Box(low=0, high=255, shape=output_shape, dtype='uint8')
        if cfg_name in reward_ranges:
            self.reward_range = get_reward_range(cfg_name, repeat)
        self.game.init()

    def close(self):
        self.game.close()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def seed(self, seed=None):
        if seed is not None:
            self.game.set_seed(seed)

    def step(self, action):
        if type(action) == np.int64:
            action_combo = self.action_combinations[action]
            # print(action_combo)
        else:
            action_combo = self.action_combinations[int(action)]
        reward = self.game.make_action(action_combo, self.repeat)
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        if state is not None:
            observation = state.screen_buffer.transpose(1, 2, 0)
            info = self._get_game_variables(state.game_variables)
        else:
            observation = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
            info = {}
        return observation, reward, done, info

    def reset(self):
        # self.seed(seed)
        self.game.new_episode()
        return self.game.get_state().screen_buffer.transpose(1, 2, 0)

    def render(self, mode='human', close=False):
        if close:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            return
        img = None
        state = self.game.get_state()
        if state is not None:
            img = state.screen_buffer
        if img is None:
            # at the end of the episode
            img = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
        if mode == 'rgb_array':
            return img
        elif mode is 'human':
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(img.transpose(1, 2, 0))

    def _get_game_variables(self, state_variables):
        info = {}
        if state_variables is not None:
            for ind, state_variable  in enumerate(state_variables):
                state_var = self.game.get_available_game_variables()[ind]
                state_var_name = str(state_var).split('.')[1]
                info[state_var_name] = state_variable
        return info
