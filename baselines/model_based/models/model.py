import torch
import torch.nn as nn
import torch.distributions as dist
from .modules.networks import FullyConnectedNetwork, ConvolutionalNetwork
from .modules.variables.latent_variables import FullyConnectedLatentVariable
from .modules.variables.observed_variables import ConvolutionalObservedVariable, FullyConnectedObservedVariable


class Model(nn.Module):
    """
    Generative Model-Based Agent
    """
    def __init__(self, n_state_variables, n_action_variables, state_prior_params,
                 action_prior_params, obs_likelihood_params, reward_likelihood_params,
                 state_inference_params, action_inference_params):
        super(Model, self).__init__()

        # networks
        self.state_prior_model = FullyConnectedNetwork(**state_prior_params)
        self.action_prior_model = FullyConnectedNetwork(**action_prior_params)
        self.obs_likelihood_model = ConvolutionalNetwork(**obs_likelihood_params)
        self.reward_likelihood_model = FullyConnectedNetwork(**reward_likelihood_params)
        self.state_inference_model = FullyConnectedNetwork(**state_inference_params)
        self.action_inference_model = FullyConnectedNetwork(**action_inference_params)

        # variables
        self.state_variable = FullyConnectedLatentVariable(prior_dist='Normal',
                                                           approx_post_dist='Normal',
                                                           n_variables=n_state_variables,
                                                           n_input=(self.state_inference_model.n_out, self.state_prior_model.n_out))
        self.action_variable = FullyConnectedLatentVariable(prior_dist= '',
                                                            approx_post_dist= '',
                                                            n_variables=n_action_variables,
                                                            n_input=(self.action_inference_model.n_out, self.action_prior_model.n_out))
        self.observation_variable = ConvolutionalObservedVariable(dist='Normal',
                                                                  n_variables= ,
                                                                  n_input=self.obs_likelihood_model.n_out)
        self.reward_variable = FullyConnectedObservedVariable(dist='Normal',
                                                              n_variables=1,
                                                              n_input=self.reward_likelihood_model.n_out)

    def act(self, observation, reward):
        # main function for interaction
        # evaluate previous reward likelihood?

        # state inference
        self.state_inference(observation)

        # form action prior, action inference
        self.step_action()
        self.action_inference()

        # reward prediction
        self.generate_reward()

        # state prediction
        self.step_state()

        # return the sampled action
        return self.action_variable.sample()

    def state_inference(self, observation):
        # infer the approx. posterior on the state
        self.state_variable.init_approx_post()
        for _ in range(self.n_inf_iter):
            # evaluate conditional likelihood of observation and state KL
            self.generate_observation()
            obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation)
            state_kl = self.state_variable.kl_divergence()
            (obs_log_likelihood + state_kl).backward(retain_graph=True)
            # update approx. posterior
            inf_input = self.state_variable.grads_and_params()
            inf_input = self.state_inference_model(inf_input)
            self.state_variable.infer(inf_input)

    def action_inference(self):
        # infer the approx. posterior on the action
        self.action_variable.init_approx_post()
        # TODO: implement planning inference

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

    def generate(self):
        # generate conditional likelihoods for the observation and reward
        self.generate_observation()
        self.generate_reward()

    def reset(self):
        # reset the variables
        self.state_variable.reset()
        self.action_variable.reset()
        self.observed_variable.reset()
        self.reward_variable.reset()

    def free_energy(self, observation, reward):
        # conditional log likelihoods
        observation_log_likelihood = self.observation_variable.cond_log_likelihood(observation)
        reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward)
        optimality_log_likelihood = reward
        conditional_log_likelihood = observation_log_likelihood + reward_log_likelihood + optimality_log_likelihood

        # kl divergences
        state_kl_divergence = self.state_variable.kl_divergence()
        action_kl_divergence = self.action_variable.kl_divergence()
        kl_divergence = state_kl_divergence + action_kl_divergence

        free_energy = kl_divergence - conditional_log_likelihood

        return free_energy

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
