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
                 reward_likelihood_args, state_inference_args, action_inference_args):
        super(Model, self).__init__()

        # networks
        self.state_prior_model = get_network(state_prior_args)
        self.action_prior_model = get_network(action_prior_args)
        self.obs_likelihood_model = get_network(obs_likelihood_args)
        self.reward_likelihood_model = get_network(reward_likelihood_args)
        self.state_inference_model = get_network(state_inference_args)
        self.action_inference_model = get_network(action_inference_args)

        # variables
        state_variable_args['n_input'] = (self.state_inference_model.n_out,
                                          self.state_prior_model.n_out)
        self.state_variable = get_variable(latent=True, args=state_variable_args)

        action_variable_args['n_input'] = (self.action_inference_model.n_out,
                                           self.action_prior_model.n_out)
        self.action_variable = get_variable(latent=True, args=action_variable_args)

        observation_variable_args['n_input'] = self.obs_likelihood_model.n_out
        self.observation_variable = get_variable(latent=False, args=observation_variable_args)

        reward_variable_args['n_input'] = self.reward_likelihood_model.n_out
        self.reward_variable = get_variable(latent=False, args=reward_variable_args)

        # miscellaneous
        self.n_inf_iter = 1
        self.training = False

    def act(self, observation, reward=None):
        self.generate_reward()
        self.step_state()
        self.state_inference(observation)
        self.generate_observation()
        self.step_action()
        self.action_inference()
        free_energy = self.free_energy(observation, reward)
        if self.training:
            free_energy.backward(retain_graph=True)
        action = one_hot_to_index(self.action_variable.sample()).cpu().numpy()
        return action, free_energy

    def state_inference(self, observation):
        self.inference_mode()
        # infer the approx. posterior on the state
        self.state_variable.init_approx_post()
        for _ in range(self.n_inf_iter):
            # evaluate conditional log likelihood of observation and state KL divergence
            self.generate_observation()
            obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum()
            state_kl = self.state_variable.kl_divergence().sum()
            (state_kl - obs_log_likelihood).backward(retain_graph=True)
            # update approx. posterior
            inf_input = self.state_variable.params_and_grads()
            inf_input = self.state_inference_model(inf_input)
            self.state_variable.infer(inf_input)
        # final evaluation
        self.generate_observation()
        obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum()
        state_kl = self.state_variable.kl_divergence().sum()
        (state_kl - obs_log_likelihood).backward(retain_graph=True)
        clear_gradients(self.generative_parameters())
        self.generative_mode()

    def action_inference(self):
        self.inference_mode()
        # infer the approx. posterior on the action
        self.action_variable.init_approx_post()
        # TODO: implement planning inference

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
        likelihood_input = self.obs_likelihood_model(state)
        self.observation_variable.generate(likelihood_input)

    def generate_reward(self):
        # generate the conditional likelihood for the reward
        state = self.state_variable.sample()
        action = self.action_variable.sample()
        likelihood_input = self.reward_likelihood_model(torch.cat((state, action), dim=1))
        self.reward_variable.generate(likelihood_input)

    def free_energy(self, observation, reward):
        # conditional log likelihoods
        observation_log_likelihood = self.observation_variable.cond_log_likelihood(observation)
        reward_log_likelihood = optimality_log_likelihood = 0.
        if reward is not None:
            reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward)
            optimality_log_likelihood = reward
        conditional_log_likelihood = observation_log_likelihood + reward_log_likelihood + optimality_log_likelihood

        # kl divergences
        state_kl_divergence = self.state_variable.kl_divergence()
        action_kl_divergence = self.action_variable.kl_divergence()
        kl_divergence = state_kl_divergence + action_kl_divergence

        free_energy = kl_divergence.sum() - conditional_log_likelihood.sum()

        return free_energy

    def reset(self):
        # reset the variables
        self.state_variable.reset()
        self.action_variable.reset()
        self.observed_variable.reset()
        self.reward_variable.reset()

        # reset the networks
        self.state_prior_model.reset()
        self.action_prior_model.reset()
        self.obs_likelihood_model.reset()
        self.reward_likelihood_model.reset()

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
