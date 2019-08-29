import torch
import torch.nn as nn
from torch import optim
from .agent import Agent
from modules.models import get_model
from modules.variables import get_variable
from misc import clear_gradients
from misc import retrace
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
        # if self.action_inference_model is not None or self._mode == 'eval':
        if True:
            # initialize the planning distributions
            self.planning_mode()
            # copy the Q-value models
            q_value_models = copy.deepcopy(self.q_value_models)
            q_value_variables = copy.deepcopy(self.q_value_variables)
            # keep track of estimated objective
            estimated_objectives = []
            if self.action_inference_model is None:
                # use gradient-based optimizer
                dist_params = self.action_variable.approx_post.get_dist_params()
                params = [param for _, param in dist_params.items()]
                act_opt = optim.Adam(params, lr=1e-4)
                act_opt.zero_grad()
            # inference iterations
            for inf_iter in range(self.n_inf_iter['action'] + 1):
                self.planning_mode()
                # sample and evaluate log probs of initial actions
                act = self.action_variable.sample()
                obs = observation.repeat(self.n_planning_samples, 1)
                kl = self.alpha['action'] * self.action_variable.kl_divergence().sum(dim=1, keepdim=True)
                estimated_objective = - kl.view(-1, 1, 1).repeat(1, self.n_planning_samples, 1)
                self.observation_variable.cond_likelihood.set_prev_obs(obs)
                # estimate the initial Q-value
                # q_value_input = [model(observation=obs, action=act) for model in q_value_models]
                # q_values = [variable(inp) for variable, inp in zip(q_value_variables, q_value_input)]
                # q_value = torch.min(q_values[0], q_values[1])
                # roll out the model
                rewards = []
                q_values_list = []
                for rollout_iter in range(self.rollout_length):
                    # generate state and reward
                    self.generate_observation(obs, act)
                    self.generate_reward(obs, act)
                    # step the action
                    reward = self.reward_variable.sample()
                    rewards.append(reward.unsqueeze(0))
                    q_value_input = [model(observation=obs, action=act) for model in q_value_models]
                    q_values = [variable(inp) for variable, inp in zip(q_value_variables, q_value_input)]
                    q_value = torch.min(q_values[0], q_values[1])
                    q_values_list.append(q_value.unsqueeze(0))

                    obs = self.observation_variable.sample()
                    self.step_action(obs)
                    act = self.action_variable.sample()

                    # estimate the Q-value
                    # q_value_input = [model(observation=obs, action=act) for model in q_value_models]
                    # q_values = [variable(inp) for variable, inp in zip(q_value_variables, q_value_input)]
                    # q_value = torch.min(q_values[0], q_values[1])

                    # add new terms to the total estimate
                    estimated_objective = estimated_objective + (self.reward_discount ** rollout_iter) * reward.view(-1, self.n_planning_samples, 1)

                q_value_input = [model(observation=obs, action=act) for model in q_value_models]
                q_values = [variable(inp) for variable, inp in zip(q_value_variables, q_value_input)]
                q_value = torch.min(q_values[0], q_values[1])
                q_values_list.append(q_value.unsqueeze(0))
                rewards = torch.cat(rewards, 0)
                q_values_list = torch.cat(q_values_list, 0)
                # estimate the final Q-value
                #old_estimated_objective = estimated_objective + (self.reward_discount ** self.n_inf_iter['action']) * q_value.view(-1, self.n_planning_samples, 1)
                # TODO: lambda as an hyper parameter
                LAMBDA = 0.75
                estimated_objective = retrace(q_values_list, rewards, None, discount=self.reward_discount, l=LAMBDA)

                # estimate and apply the gradients
                objective = - estimated_objective
                # average over samples, sum over other dimensions
                objective.mean(dim=1).sum().backward(retain_graph=True)

                if inf_iter < self.n_inf_iter['action']:
                    if self.action_inference_model is not None:
                        # update the approximate posterior using the inference model
                        params, grads = self.action_variable.params_and_grads()
                        inf_input = self.action_inference_model(params=params, grads=grads, observation=observation)
                        self.action_variable.infer(inf_input)
                    else:
                        # update the approximate posterior using gradient-based optimizer
                        act_opt.step()

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
        self.generate_observation(self._prev_obs, self._prev_action)
        self.generate_reward(self._prev_obs, self._prev_action)
        self.observation_variable.cond_likelihood.set_prev_obs(observation)

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