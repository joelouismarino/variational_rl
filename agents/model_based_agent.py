import torch
import torch.nn as nn
from .agent import Agent
from modules.models import get_model
from modules.variables import get_variable
from misc import clear_gradients
import copy


class ModelBasedAgent(Agent):
    """
    Model-Based Variational RL Agent.
    """
    def __init__(self, action_variable_args, action_prior_args,
                 action_inference_args, observation_variable_args, obs_likelihood_args,
                 reward_variable_args, reward_likelihood_args, q_value_model_args,
                 misc_args):
        super(ModelBasedAgent, self).__init__(misc_args)

        # models
        self.action_prior_model = get_model(action_prior_args)
        self.action_inference_model = get_model(action_inference_args)
        self.reward_likelihood_model = get_model(reward_likelihood_args)
        self.obs_likelihood_model = get_model(obs_likelihood_args)
        self.q_value_models = nn.ModuleList([get_model(copy.deepcopy(q_value_model_args)) for _ in range(2)])
        self.target_q_value_models = nn.ModuleList([get_model(copy.deepcopy(q_value_model_args)) for _ in range(2)])

        # variables
        action_variable_args['n_input'] = [None, None]
        if self.action_prior_model is not None:
            action_variable_args['n_input'][0] = self.action_prior_model.n_out
        if self.action_inference_model is not None:
           action_variable_args['n_input'][1] = self.action_inference_model.n_out
        self.action_variable = get_variable(type='latent', args=action_variable_args)

        observation_variable_args['n_input'] = self.obs_likelihood_model.n_out
        self.observation_variable = get_variable(type='observed', args=observation_variable_args)

        reward_variable_args['n_input'] = self.reward_likelihood_model.n_out
        self.reward_variable = get_variable(type='observed', args=reward_variable_args)

        if self.q_value_models is not None:
            self.q_value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': self.q_value_models[0].n_out}) for _ in range(2)])
            self.target_q_value_variables = nn.ModuleList([get_variable(type='value', args={'n_input': self.q_value_models[0].n_out}) for _ in range(2)])

        if self.action_variable.approx_post.update == 'iterative':
            self.rollout_length = misc_args['max_rollout_length']
            self.n_planning_samples = misc_args['n_planning_samples']
            self.n_inf_iter = {'action': misc_args['n_inf_iter']['action']}

    def step_action(self, observation, **kwargs):
        """
        Generate the prior on the action.
        """
        self.action_variable.generative_mode()
        if self.action_prior_model is not None:
            if not self.action_variable.reinitialized:
                if self._prev_action is not None:
                    action = self._prev_action
                else:
                    action = self.action_variable.sample()
                if self.obs_normalizer:
                    observation = self.obs_normalizer(observation, update=self._mode=='eval')
                prior_input = self.action_prior_model(observation=observation, action=action)
                self.action_variable.step(prior_input)

    def action_inference(self, observation, action=None,**kwargs):
        """
        Infer the approximate posterior on the action.
        """
        self.action_variable.inference_mode()
        if self.action_inference_model is not None:
            # initialize the planning distributions
            self.planning_mode()
            # copy the Q-value models
            q_value_models = copy.deepcopy(self.q_value_models)
            q_value_variables = copy.deepcopy(self.q_value_variables)
            # keep track of estimated objective
            estimated_objectives = []
            # inference iterations
            for inf_iter in range(self.n_inf_iter['action'] + 1):
                self.planning_mode()
                # sample and evaluate log probs of initial actions
                action = self.action_variable.sample()
                obs = observation.repeat(self.n_planning_samples, 1)
                # TODO: this should be the KL divergence at the first step
                kl = self.alpha['action'] * self.action_variable.kl_divergence().sum(dim=1, keepdim=True)
                estimated_objective = - kl.view(-1, 1, 1).repeat(1, self.n_planning_samples, 1)
                # estimate the initial Q-value
                # q_value_input = [model(observation=obs, action=action) for model in q_value_models]
                # q_values = [variable(inp) for variable, inp in zip(q_value_variables, q_value_input)]
                # q_value = torch.min(q_values[0], q_values[1])
                # roll out the model
                for rollout_iter in range(self.rollout_length):
                    # generate state and reward
                    self.generate_observation(obs, action)
                    self.generate_reward(obs, action)
                    # step the action
                    obs = self.observation_variable.sample()
                    reward = self.reward_variable.sample()
                    self.step_action(obs)
                    action = self.action_variable.sample()

                    # estimate the Q-value
                    # q_value_input = [model(observation=obs, action=action) for model in q_value_models]
                    # q_values = [variable(inp) for variable, inp in zip(q_value_variables, q_value_input)]
                    # q_value = torch.min(q_values[0], q_values[1])

                    # add new terms to the total estimate
                    estimated_objective = estimated_objective + (self.reward_discount ** rollout_iter) * reward.view(-1, self.n_planning_samples, 1)

                # estimate the final Q-value
                q_value_input = [model(observation=obs, action=action) for model in q_value_models]
                q_values = [variable(inp) for variable, inp in zip(q_value_variables, q_value_input)]
                q_value = torch.min(q_values[0], q_values[1])
                estimated_objective = estimated_objective + (self.reward_discount ** self.n_inf_iter['action']) * q_value.view(-1, self.n_planning_samples, 1)

                # estimate and apply the gradients
                objective = - estimated_objective
                # average over samples, sum over other dimensions
                objective.mean(dim=1).sum().backward(retain_graph=True)

                if inf_iter < self.n_inf_iter['action']:
                    # update the approximate posterior using the inference model
                    params, grads = self.action_variable.params_and_grads()
                    inf_input = self.action_inference_model(params=params, grads=grads)
                    self.action_variable.infer(inf_input)

                # store the length of the planning rollout and objective estimate
                if self._mode == 'train':
                    estimated_objectives.append(estimated_objective.detach().mean(dim=1))

            # save the maximum rollout length, averaged over inference iterations
            if self._mode == 'train':
                estimated_objectives = torch.stack(estimated_objectives)
                inference_improvement = - estimated_objectives[0] + estimated_objectives[-1]
                self.collector.inference_improvement['action'].append(inference_improvement)

            self.acting_mode()

        clear_gradients(self.generative_parameters())
        self.generative_mode()
        # predict next observation and reward
        action = self.action_variable.sample()
        self.generate_observation(observation, action)
        self.generate_reward(observation, action)

    def generate_reward(self, observation, action, **kwargs):
        """
        Generate the conditional likelihood for the reward.
        """
        likelihood_input = self.reward_likelihood_model(observation=observation, action=action)
        self.reward_variable.generate(likelihood_input)

    def generate_observation(self, observation, action, **kwargs):
        """
        Generate the conditional likelihood for the observation (state).
        """
        likelihood_input = self.obs_likelihood_model(observation=observation, action=action)
        self.observation_variable.generate(likelihood_input)

    def planning_mode(self):
        """
        Set the variables and models for action inference (planning).
        """
        self._planning = True
        self.observation_variable.planning_mode(self.n_planning_samples)
        self.action_variable.planning_mode(self.n_planning_samples)
        self.reward_variable.planning_mode(self.n_planning_samples)
        if self.action_prior_model is not None:
            self.action_prior_model.planning_mode(self.n_planning_samples)
        self.obs_likelihood_model.planning_mode(self.n_planning_samples)
        self.reward_likelihood_model.planning_mode(self.n_planning_samples)

    def acting_mode(self):
        """
        Set the variables and models for acting.
        """
        self._planning = False
        self.observation_variable.acting_mode()
        self.action_variable.acting_mode()
        self.reward_variable.acting_mode()
        if self.action_prior_model is not None:
            self.action_prior_model.acting_mode()
        self.obs_likelihood_model.acting_mode()
        self.reward_likelihood_model.acting_mode()
