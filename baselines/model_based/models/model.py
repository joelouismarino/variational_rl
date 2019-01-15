import torch
import torch.nn as nn
import torch.distributions as dist
from ..modules.networks import get_network
from ..modules.variables import get_variable
from ..misc import clear_gradients, one_hot_to_index


class Model(nn.Module):
    """
    Generative Model-Based Agent
    """
    def __init__(self, state_variable_args, action_variable_args,
                 observation_variable_args, reward_variable_args,
                 state_prior_args, action_prior_args, obs_likelihood_args,
                 reward_likelihood_args, state_inference_args,
                 action_inference_args, misc_args):
        super(Model, self).__init__()

        # networks
        self.state_prior_model = get_network(state_prior_args)
        self.action_prior_model = get_network(action_prior_args)
        self.obs_likelihood_model = get_network(obs_likelihood_args)
        self.reward_likelihood_model = get_network(reward_likelihood_args)
        self.state_inference_model = get_network(state_inference_args)
        self.action_inference_model = get_network(action_inference_args)

        # variables
        state_variable_args['n_input'] = (self.state_prior_model.n_out,
                                          self.state_inference_model.n_out)
        self.state_variable = get_variable(latent=True, args=state_variable_args)

        action_variable_args['n_input'] = (self.action_prior_model.n_out,
                                           self.action_inference_model.n_out)
        self.action_variable = get_variable(latent=True, args=action_variable_args)

        observation_variable_args['n_input'] = self.obs_likelihood_model.n_out
        self.observation_variable = get_variable(latent=False, args=observation_variable_args)

        reward_variable_args['n_input'] = self.reward_likelihood_model.n_out
        self.reward_variable = get_variable(latent=False, args=reward_variable_args)

        # miscellaneous
        self.n_inf_iter = 1
        self.training = False
        self.optimality_scale = misc_args['optimality_scale']

        self.objectives = {'observation': [], 'reward': [], 'optimality': [],
                           'state': [], 'action': []}
        self.log_probs = {'action': []}

        self.state_inf_free_energies = []
        self.obs_reconstruction = None
        self.obs_prediction = None

    def act(self, observation, reward=None):
        self.step_state()
        self.state_inference(observation, reward)
        self.generate_observation()
        self.generate_reward()
        self.step_action()
        self.action_inference()
        action = self.action_variable.sample()
        if self.training:
            self._collect_objectives_and_log_probs(observation, reward)
        return self.convert_action(action).cpu().numpy()

    def final_reward(self, reward):
        self.step_state()
        self.state_variable.init_approx_post()
        self.generate_reward()
        if self.training:
            self._collect_objectives_and_log_probs(None, reward)

    # free_energy = self.free_energy(observation, reward)
    # self.free_energies.append(free_energy)
    # free_energy.backward(retain_graph=True)
    # if reward is not None:
        # self.rewards.append(reward)
    # log_prob = self.action_variable.approx_post_dist.log_prob(action)
    # self.policy_log_probs.append(log_prob)

    # def state_inference(self, observation):
    #     self.inference_mode()
    #     # infer the approx. posterior on the state
    #     inf_input = observation - 0.5
    #     inf_input = self.state_inference_model(inf_input)
    #     self.state_variable.infer(inf_input)
    #     # final evaluation
    #     self.generate_observation()
    #     obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum()
    #     state_kl = self.state_variable.kl_divergence().sum()
    #     (state_kl - obs_log_likelihood).backward(retain_graph=True)
    #     clear_gradients(self.generative_parameters())
    #     self.generative_mode()

    def state_inference(self, observation, reward):
        self.inference_mode()
        self.state_inf_free_energies = []
        # infer the approx. posterior on the state
        self.state_variable.init_approx_post()
        for inf_iter in range(self.n_inf_iter):
            # evaluate conditional log likelihood of observation and state KL divergence
            self.generate_observation()
            if reward is not None:
                self.generate_reward()
            if inf_iter == 0:
                self.obs_prediction = self.observation_variable.likelihood_dist.loc.detach()
            obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum()
            if reward is not None:
                reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward).sum()
            state_kl = self.state_variable.kl_divergence().sum()
            state_inf_free_energy = state_kl - obs_log_likelihood
            if reward is not None:
                state_inf_free_energy -= reward_log_likelihood
            self.state_inf_free_energies.append(state_inf_free_energy)
            (state_inf_free_energy).backward(retain_graph=True)
            # update approx. posterior
            inf_input = self.state_variable.params_and_grads()
            inf_input = self.state_inference_model(inf_input)
            self.state_variable.infer(inf_input)
        # final evaluation
        self.generate_observation()
        if reward is not None:
            self.generate_reward()
        obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum()
        if reward is not None:
            reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward).sum()
        state_kl = self.state_variable.kl_divergence().sum()
        state_inf_free_energy = state_kl - obs_log_likelihood
        if reward is not None:
            state_inf_free_energy -= reward_log_likelihood
        self.state_inf_free_energies.append(state_inf_free_energy)
        (state_inf_free_energy).backward(retain_graph=True)
        clear_gradients(self.generative_parameters())
        self.generative_mode()
        self.obs_reconstruction = self.observation_variable.likelihood_dist.loc.detach()

    def action_inference(self):
        self.inference_mode()
        # infer the approx. posterior on the action
        # self.action_variable.init_approx_post()
        # TODO: implement planning inference

        state = self.state_variable.sample()
        action = self.action_variable.sample()
        inf_input = self.action_prior_model(torch.cat((state, action), dim=1))
        self.action_variable.infer(inf_input)

        clear_gradients(self.generative_parameters())
        self.generative_mode()

    def step_state(self):
        # calculate the prior on the state variable
        state = self.state_variable.sample()
        action = self.action_variable.sample()
        prior_input = self.state_prior_model(torch.cat((state, action), dim=1))
        self.state_variable.step(prior_input)

    def step_action(self):
        # calculate the prior on the action variable
        state = self.state_variable.sample()
        action = self.action_variable.sample()
        prior_input = self.action_prior_model(torch.cat((state, action), dim=1))
        self.action_variable.step(prior_input)

    def generate_observation(self):
        # generate the conditional likelihood for the observation
        state = self.state_variable.sample()
        action = self.action_variable.sample()
        # likelihood_input = self.obs_likelihood_model(state)
        likelihood_input = self.obs_likelihood_model(torch.cat((state, action), dim=1))
        self.observation_variable.generate(likelihood_input)

    def generate_reward(self):
        # generate the conditional likelihood for the reward
        state = self.state_variable.sample()
        action = self.action_variable.sample()
        likelihood_input = self.reward_likelihood_model(torch.cat((state, action), dim=1))
        # likelihood_input = self.reward_likelihood_model(state)
        self.reward_variable.generate(likelihood_input)

    def free_energy(self, observation, reward):
        cond_log_likelihood = self.cond_log_likelihood(observation, reward)
        kl_divergence = self.kl_divergence()
        free_energy = kl_divergence - cond_log_likelihood
        return free_energy

    def cond_log_likelihood(self, observation, reward):
        obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum()
        reward_log_likelihood = opt_log_likelihood = 0.
        if reward is not None:
            reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward).sum()
            opt_log_likelihood = self.optimality_scale * (reward - 1.)
        cond_log_likelihood = obs_log_likelihood + reward_log_likelihood + opt_log_likelihood
        return cond_log_likelihood

    def kl_divergence(self):
        state_kl_divergence = self.state_variable.kl_divergence().sum()
        action_kl_divergence = self.action_variable.kl_divergence()
        kl_divergence = state_kl_divergence + action_kl_divergence
        return kl_divergence

    def _collect_objectives_and_log_probs(self, observation, reward):
        if reward is None:
            # beginning of an episode
            self.objectives['reward'].append(torch.tensor(0.))
            self.objectives['optimality'].append(torch.tensor(0.))

            self.log_probs['action'].append(torch.tensor(0.))

        if observation is not None:
            # beginning of an episode or during an episode
            self.objectives['observation'].append(-self.observation_variable.cond_log_likelihood(observation).sum())
            self.objectives['state'].append(self.state_variable.kl_divergence().sum())
            self.objectives['action'].append(self.action_variable.kl_divergence().sum())

            action = self.convert_action(self.action_variable.sample())
            self.log_probs['action'].append(self.action_variable.approx_post_dist.log_prob(action).sum())

        if reward is not None:
            # during an episode or end of an episode
            # TODO: subtracting 1 is a hack, need to rescale reward
            self.objectives['reward'].append(-self.reward_variable.cond_log_likelihood(reward).sum())
            optimality = torch.tensor(self.optimality_scale * (reward - 1.))
            self.objectives['optimality'].append(-optimality)

        if observation is None:
            # end of an episode
            self.objectives['observation'].append(torch.tensor(0.))
            self.objectives['state'].append(torch.tensor(0.))
            self.objectives['action'].append(torch.tensor(0.))

    def convert_action(self, action):
        # converts categorical action from one-hot encoding to the action index
        if self.action_variable.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
            action = one_hot_to_index(action)
        return action

    def reset(self):
        # reset the variables
        self.state_variable.reset()
        self.action_variable.reset()
        self.observation_variable.reset()
        self.reward_variable.reset()

        # reset the networks
        self.state_prior_model.reset()
        self.action_prior_model.reset()
        self.obs_likelihood_model.reset()
        self.reward_likelihood_model.reset()

        # reset the objectives and log probs
        self.objectives = {'observation': [], 'reward': [], 'optimality': [],
                           'state': [], 'action': []}
        self.log_probs = {'action': []}

    @property
    def device(self):
        return self.inference_parameters()[0].device

    def parameters(self):
        param_dict = {}

        param_dict['state_inference_model'] = nn.ParameterList()
        param_dict['state_inference_model'].extend(list(self.state_inference_model.parameters()))
        param_dict['state_inference_model'].extend(list(self.state_variable.inference_parameters()))

        param_dict['action_inference_model'] = nn.ParameterList()
        param_dict['action_inference_model'].extend(list(self.action_inference_model.parameters()))
        param_dict['action_inference_model'].extend(list(self.action_variable.inference_parameters()))

        param_dict['state_prior_model'] = nn.ParameterList()
        param_dict['state_prior_model'].extend(list(self.state_prior_model.parameters()))
        param_dict['state_prior_model'].extend(list(self.state_variable.generative_parameters()))

        param_dict['action_prior_model'] = nn.ParameterList()
        param_dict['action_prior_model'].extend(list(self.action_prior_model.parameters()))
        param_dict['action_prior_model'].extend(list(self.action_variable.generative_parameters()))

        param_dict['obs_likelihood_model'] = nn.ParameterList()
        param_dict['obs_likelihood_model'].extend(list(self.obs_likelihood_model.parameters()))
        param_dict['obs_likelihood_model'].extend(list(self.observation_variable.parameters()))

        param_dict['reward_likelihood_model'] = nn.ParameterList()
        param_dict['reward_likelihood_model'].extend(list(self.reward_likelihood_model.parameters()))
        param_dict['reward_likelihood_model'].extend(list(self.reward_variable.parameters()))

        return param_dict

    def inference_parameters(self):
        params = nn.ParameterList()
        params.extend(list(self.state_inference_model.parameters()))
        params.extend(list(self.action_inference_model.parameters()))
        params.extend(list(self.state_variable.inference_parameters()))
        params.extend(list(self.action_variable.inference_parameters()))
        return params

    def generative_parameters(self):
        params = nn.ParameterList()
        params.extend(list(self.state_prior_model.parameters()))
        params.extend(list(self.action_prior_model.parameters()))
        params.extend(list(self.obs_likelihood_model.parameters()))
        params.extend(list(self.reward_likelihood_model.parameters()))
        params.extend(list(self.state_variable.generative_parameters()))
        params.extend(list(self.action_variable.generative_parameters()))
        params.extend(list(self.observation_variable.parameters()))
        params.extend(list(self.reward_variable.parameters()))
        return params

    def inference_mode(self):
        self.state_variable.inference_mode()
        self.action_variable.inference_mode()
        self.obs_likelihood_model.detach_hidden_state()
        self.reward_likelihood_model.detach_hidden_state()

    def generative_mode(self):
        self.state_variable.generative_mode()
        self.action_variable.generative_mode()
        self.obs_likelihood_model.attach_hidden_state()
        self.reward_likelihood_model.attach_hidden_state()
