import torch
import torch.nn as nn
import torch.distributions as dist
from .agent import Agent
from ..modules.models import get_model
from ..modules.variables import get_variable
from ..misc import clear_gradients, one_hot_to_index
from ..util.normalization_util import Normalizer


class GenerativeAgent(Agent):
    """
    Variational RL Agent with Generative State Estimation
    """
    def __init__(self, state_variable_args, action_variable_args,
                 observation_variable_args, reward_variable_args,
                 done_variable_args, state_prior_args, action_prior_args,
                 obs_likelihood_args, reward_likelihood_args,
                 done_likelihood_args, state_inference_args,
                 action_inference_args, value_model_args, misc_args):
        super(GenerativeAgent, self).__init__()

        # models
        self.state_prior_model = get_model('generative', 'state', 'prior', state_prior_args)
        self.action_prior_model = get_model('generative', 'action', 'prior', action_prior_args)
        self.obs_likelihood_model = get_model('generative', 'observation', 'likelihood', obs_likelihood_args)
        self.reward_likelihood_model = get_model('generative', 'reward', 'likelihood', reward_likelihood_args)
        self.done_likelihood_model = get_model('generative', 'done', 'likelihood', done_likelihood_args)
        self.state_inference_model = get_model('generative', 'state', 'inference', state_inference_args)
        self.action_inference_model = get_model('generative', 'action', 'inference', action_inference_args)
        self.value_model = get_model('value', value_model_args)

        # variables
        state_variable_args['n_input'] = [None, None]
        if self.state_prior_model is not None:
            state_variable_args['n_input'][0] = self.state_prior_model.n_out
        if self.state_inference_model is not None:
            state_variable_args['n_input'][1] = self.state_inference_model.n_out
        self.state_variable = get_variable(type='latent', args=state_variable_args)

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

        done_variable_args['n_input'] = self.done_likelihood_model.n_out
        self.done_variable = get_variable(type='observed', args=done_variable_args)

        if self.value_model is not None:
            self.value_variable = get_variable(type='value', args={'n_input': self.value_model.n_out})

        # miscellaneous
        self.optimality_scale = misc_args['optimality_scale']
        self.n_state_samples = misc_args['n_state_samples']
        self.n_inf_iter = {'state': misc_args['n_inf_iter']['state'],
                           'action': misc_args['n_inf_iter']['action']}
        self.kl_min = {'state': misc_args['kl_min']['state'],
                       'action': misc_args['kl_min']['action']}
        self.kl_min_anneal_rate = {'state': misc_args['kl_min_anneal_rate']['state'],
                                   'action': misc_args['kl_min_anneal_rate']['action']}
        self.kl_factor = {'state': misc_args['kl_factor']['state'],
                          'action': misc_args['kl_factor']['action']}
        self.kl_factor_anneal_rate = {'state': misc_args['kl_factor_anneal_rate']['state'],
                                      'action': misc_args['kl_factor_anneal_rate']['action']}
        if self.n_inf_iter['action'] > 0:
            self.n_planning_samples = misc_args['n_planning_samples']
            self.max_rollout_length = misc_args['max_rollout_length']

        # mode (either 'train' or 'eval')
        self._mode = 'train'

        # stores the variables during an episode
        self.episode = {'observation': [], 'reward': [], 'done': [],
                        'state': [], 'action': [], 'reconstruction': [],
                        'prediction': []}
        # stores the objectives during an episode
        self.objectives = {'observation': [], 'reward': [], 'optimality': [],
                           'done': [], 'state': [], 'action': [], 'value': []}
        # stores inference improvement
        self.inference_improvement = {'state': [], 'action': []}
        # stores planning rollout lengths during training
        self.rollout_lengths = []
        # stores the log probabilities during an episode
        self.log_probs = {'action': []}
        # store the values during training
        self.values = []

        self.valid = []
        self._prev_action = None
        self.batch_size = 1
        self.gae_lambda = misc_args['gae_lambda']
        self.reward_discount = misc_args['reward_discount']
        self._planning = False

    def state_inference(self, observation, reward, done, valid, **kwargs):
        # infer the approx. posterior on the state
        if self.state_inference_model is not None:
            self.inference_mode()
            self.state_variable.init_approx_post()
            for inf_iter in range(self.n_inf_iter['state']):
                # evaluate conditional log likelihood of observation and state KL divergence
                self.generate_observation()
                self.generate_reward()
                self.generate_done()
                obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation)
                obs_log_likelihood = obs_log_likelihood.view(self.n_state_samples, -1, 1).mean(dim=0)
                reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward)
                reward_log_likelihood = reward_log_likelihood.view(self.n_state_samples, -1, 1).mean(dim=0)
                done_log_likelihood = self.done_variable.cond_log_likelihood(done)
                done_log_likelihood = done_log_likelihood.view(self.n_state_samples, -1, 1).mean(dim=0)
                state_kl = self.state_variable.kl_divergence().sum(dim=1, keepdim=True)
                state_inf_free_energy = state_kl - (1 - done) * obs_log_likelihood - reward_log_likelihood - done_log_likelihood
                state_inf_free_energy = valid * state_inf_free_energy
                if inf_iter == 0:
                    initial_free_energy = state_inf_free_energy
                    #save the predictions for marginal likelihood estimation
                    self.observation_variable.save_prediction()
                    self.reward_variable.save_prediction()
                    self.done_variable.save_prediction()

                clamped_state_kl = torch.clamp(self.state_variable.kl_divergence(), min=self.kl_min['state']).sum(dim=1, keepdim=True)
                state_inf_free_energy = valid * (clamped_state_kl - (1 - done) * obs_log_likelihood - reward_log_likelihood - done_log_likelihood)
                (state_inf_free_energy.sum()).backward(retain_graph=True)
                # update approx. posterior
                params, grads = self.state_variable.params_and_grads()
                inf_input = self.state_inference_model(params, grads)
                self.state_variable.infer(inf_input)
            # final evaluation
            self.generate_observation()
            self.generate_reward()
            self.generate_done()
            obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation)
            obs_log_likelihood = obs_log_likelihood.view(self.n_state_samples, -1, 1).mean(dim=0)
            reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward)
            reward_log_likelihood = reward_log_likelihood.view(self.n_state_samples, -1, 1).mean(dim=0)
            done_log_likelihood = self.done_variable.cond_log_likelihood(done)
            done_log_likelihood = done_log_likelihood.view(self.n_state_samples, -1, 1).mean(dim=0)
            state_kl = self.state_variable.kl_divergence().sum(dim=1, keepdim=True)
            state_inf_free_energy = state_kl - (1 - done) * obs_log_likelihood - reward_log_likelihood - done_log_likelihood
            state_inf_free_energy = valid * state_inf_free_energy
            final_free_energy = state_inf_free_energy
            inference_improvement = torch.zeros(initial_free_energy.shape).to(self.device)
            valid_inds = torch.nonzero(valid[:,0])
            inference_improvement[valid_inds] = initial_free_energy[valid_inds] - final_free_energy[valid_inds]
            # inference_improvement[valid_inds] = 100 * inference_improvement[valid_inds] / initial_free_energy[valid_inds]
            self.inference_improvement['state'].append(inference_improvement)

            clamped_state_kl = torch.clamp(self.state_variable.kl_divergence(), min=self.kl_min['state']).sum(dim=1, keepdim=True)
            state_inf_free_energy = valid * (clamped_state_kl - (1 - done) * obs_log_likelihood - reward_log_likelihood - done_log_likelihood)
            (state_inf_free_energy.sum()).backward(retain_graph=True)
            clear_gradients(self.generative_parameters())
            self.generative_mode()

        self.generate_observation()
        self.generate_reward()
        self.generate_done()

    def action_inference(self, done, valid, action=None, **kwargs):
        if self.action_inference_model is not None:
            self.inference_mode()
            # infer the approx. posterior on the action
            self.action_variable.init_approx_post()
            if self.action_variable.inference_type == 'direct':
                # model-free action inference
                state = self.state_variable.sample()
                # hidden_state = self.state_prior_model.network.state
                if self._prev_action is not None:
                    action = self._prev_action
                else:
                    action = self.action_variable.sample()
                # inf_input = self.action_inference_model(state, action, hidden_state)
                inf_input = self.action_inference_model(state, action)
                self.action_variable.infer(inf_input)
            else:
                # model-based action inference
                # keep track of rollout length and estimated return
                rollout_lengths = []
                estimated_returns = []
                # initialize the planning distributions
                self.planning_mode()

                # if we can't backprop through action sampling
                if not self.action_variable.approx_post_dist.has_rsample:
                    # estimate the current value as baseline for REINFORCE gradients
                    current_done = done.repeat(self.n_planning_samples, 1)
                    current_value = self.estimate_value(done=current_done).detach()
                    current_value = current_value.view(-1, self.n_planning_samples, 1)

                # inference iterations
                for inf_iter in range(self.n_inf_iter['action'] + 1):

                    # initialize the planning distributions
                    self.planning_mode()

                    # sample and evaluate log probs of initial actions
                    action = self.action_variable.sample()
                    action_ind = self._convert_action(action)
                    original_batch_shape = self.action_variable.approx_post_dist.batch_shape[0]
                    # expanded_action_dist = self.action_variable.approx_post_dist.expand([self.n_planning_samples, original_batch_shape])
                    # action_log_prob = expanded_action_dist.log_prob(action_ind.view(self.n_planning_samples, -1)).view(-1, self.n_planning_samples, 1)

                    estimated_return = 0.

                    total_flag = True
                    cumulative_flag = None
                    rollout_iter = 0
                    # roll out the model
                    for rollout_iter in range(self.max_rollout_length):
                        # step the state
                        self.step_state()
                        # generate observation, reward, and done
                        self.generate_observation()
                        self.generate_reward()
                        self.generate_done()
                        # step the action
                        self.step_action()

                        # evaluate the objective
                        reward = self.reward_variable.sample()
                        optimality_log_likelihood = reward
                        # TODO: evaluate other terms
                        # reward_mi = self.reward_variable.info_gain()
                        # observation_mi = self.observation_variable.info_gain()
                        # done_mi = self.done_variable.info_gain()
                        #
                        # optimality_log_likelihood = self.optimality_scale * (reward - 1.)
                        #
                        # value = self.estimate_value()

                        # evaluate done variables to determine whether all rollouts are completed
                        done = self.done_variable.sample().view(-1, self.n_planning_samples, 1)
                        if cumulative_flag is None:
                            cumulative_flag = 1 - done
                        cumulative_flag = cumulative_flag * (1 - done)
                        total_flag = cumulative_flag.sum().sign().item()

                        # add new terms to the total estimate
                        new_terms = cumulative_flag * optimality_log_likelihood.view(-1, self.n_planning_samples, 1)
                        estimated_return = estimated_return + new_terms

                        # exit if all sampled rollouts are done
                        if not total_flag:
                            break

                    # estimate and apply the policy gradients
                    if not self.action_variable.approx_post_dist.has_rsample:
                        advantages = estimated_return - current_value
                        objective = - action_log_prob * advantages.detach()
                    else:
                        objective = - estimated_return
                    # average over samples, sum over other dimensions
                    objective.mean(dim=1).sum().backward(retain_graph=True)

                    if inf_iter < self.n_inf_iter['action']:
                        # update the approximate posterior using the inference model
                        params, grads = self.action_variable.params_and_grads()
                        inf_input = self.action_inference_model(params, grads)
                        self.action_variable.infer(inf_input)

                    # store the length of the planning rollout
                    if self._mode == 'train':
                        rollout_lengths.append(rollout_iter)
                        estimated_returns.append(estimated_return.detach().mean(dim=1))

                # save the maximum rollout length, averaged over inference iterations
                if self._mode == 'train':
                    ave_rollout_length = sum(rollout_lengths) / len(rollout_lengths)
                    self.rollout_lengths.append(ave_rollout_length)
                    # TODO: need to only collect inference improvement for valid steps
                    estimated_returns = torch.stack(estimated_returns)
                    inference_improvement = - estimated_returns[0] + estimated_returns[-1]
                    self.inference_improvement['action'].append(inference_improvement)

                self.acting_mode()

            clear_gradients(self.generative_parameters())
            self.generative_mode()

    def step_state(self, **kwargs):
        # calculate the prior on the state variable
        if self.state_prior_model is not None:
            if not self.state_variable.reinitialized:
                state = self.state_variable.sample()
                if self._prev_action is not None and not self._planning:
                    action = self._prev_action
                else:
                    action = self.action_variable.sample()
                prior_input = self.state_prior_model(state, action)
                self.state_variable.step(prior_input)

    def step_action(self, action=None, **kwargs):
        # calculate the prior on the action variable
        if self.action_prior_model is not None:
            if not self.action_variable.reinitialized:
                state = self.state_variable.sample()
                # hidden_state = self.state_prior_model.network.state
                if self._prev_action is not None and not self._planning:
                    action = self._prev_action
                else:
                    action = self.action_variable.sample()
                # prior_input = self.action_prior_model(state, action, hidden_state)
                prior_input = self.action_prior_model(state, action)
                self.action_variable.step(prior_input)

    def generate_observation(self):
        # generate the conditional likelihood for the observation
        state = self.state_variable.sample(n_samples=self.n_state_samples)
        # hidden_state = self.state_prior_model.network.state
        # likelihood_input = self.obs_likelihood_model(state, hidden_state)
        likelihood_input = self.obs_likelihood_model(state)
        self.observation_variable.generate(likelihood_input)

    def generate_reward(self):
        # generate the conditional likelihood for the reward
        state = self.state_variable.sample(n_samples=self.n_state_samples)
        # hidden_state = self.state_prior_model.network.state
        # likelihood_input = self.reward_likelihood_model(state, hidden_state)
        likelihood_input = self.reward_likelihood_model(state)
        self.reward_variable.generate(likelihood_input)

    def generate_done(self):
        # generate the conditional likelihood for episode being done
        state = self.state_variable.sample(n_samples=self.n_state_samples)
        # hidden_state = self.state_prior_model.network.state
        # likelihood_input = self.done_likelihood_model(state, hidden_state)
        likelihood_input = self.done_likelihood_model(state)
        self.done_variable.generate(likelihood_input)

    def estimate_value(self, done, **kwargs):
        # estimate the value of the current state
        state = self.state_variable.sample()
        # hidden_state = self.state_prior_model.network.state
        # value = self.value_variable(self.value_model(state, hidden_state)) * (1 - done)
        value = self.value_variable(self.value_model(state)) * (1 - done)
        if not self._planning:
            self.values.append(value)
        return value

    def planning_mode(self):
        self._planning = True
        # set the variables and models for action inference
        self.state_variable.planning_mode(self.n_planning_samples)
        self.action_variable.planning_mode(self.n_planning_samples)
        self.observation_variable.planning_mode()
        self.reward_variable.planning_mode()
        self.done_variable.planning_mode()

        if self.state_prior_model is not None:
            self.state_prior_model.planning_mode(self.n_planning_samples)
        if self.action_prior_model is not None:
            self.action_prior_model.planning_mode(self.n_planning_samples)
        self.obs_likelihood_model.planning_mode(self.n_planning_samples)
        self.reward_likelihood_model.planning_mode(self.n_planning_samples)
        self.done_likelihood_model.planning_mode(self.n_planning_samples)

    def acting_mode(self):
        self._planning = False

        self.state_variable.acting_mode()
        self.action_variable.acting_mode()
        self.observation_variable.acting_mode()
        self.reward_variable.acting_mode()
        self.done_variable.acting_mode()

        if self.state_prior_model is not None:
            self.state_prior_model.acting_mode()
        if self.action_prior_model is not None:
            self.action_prior_model.acting_mode()
        self.obs_likelihood_model.acting_mode()
        self.reward_likelihood_model.acting_mode()
        self.done_likelihood_model.acting_mode()
