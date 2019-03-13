import torch
import torch.nn as nn
import torch.distributions as dist
from .agent import Agent
from ..modules.models import get_model
from ..modules.variables import get_variable
from ..misc import clear_gradients, one_hot_to_index


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
        self.n_inf_iter = {'state': misc_args['n_inf_iter']['state'],
                           'action': misc_args['n_inf_iter']['action']}
        self.kl_min = {'state': misc_args['kl_min']['state'],
                       'action': misc_args['kl_min']['action']}

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
        # stores the log probabilities during an episode
        self.log_probs = {'action': []}
        # store the values during training
        self.values = []

        self.valid = []
        self._prev_action = None
        self.batch_size = 1
        self.gae_lambda = misc_args['gae_lambda']

        self.obs_reconstruction = None
        self.obs_prediction = None

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
                obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum(dim=1, keepdim=True)
                reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward).sum(dim=1, keepdim=True)
                done_log_likelihood = self.done_variable.cond_log_likelihood(done).sum(dim=1, keepdim=True)
                state_kl = self.state_variable.kl_divergence().sum(dim=1, keepdim=True)
                state_inf_free_energy = state_kl - (1 - done) * obs_log_likelihood - reward_log_likelihood - done_log_likelihood
                state_inf_free_energy = valid * state_inf_free_energy
                if inf_iter == 0:
                    self.obs_prediction = self.observation_variable.likelihood_dist.loc.detach()
                    if len(self.obs_prediction.shape) != len(observation.shape):
                        self.obs_prediction = self.obs_prediction.view(observation.shape)
                    initial_free_energy = state_inf_free_energy

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
            obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum(dim=1, keepdim=True)
            reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward).sum(dim=1, keepdim=True)
            done_log_likelihood = self.done_variable.cond_log_likelihood(done).sum(dim=1, keepdim=True)
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
            self.obs_reconstruction = self.observation_variable.likelihood_dist.loc.detach()
            if len(self.obs_reconstruction.shape) != len(observation.shape):
                self.obs_reconstruction = self.obs_reconstruction.view(observation.shape)
        else:
            self.generate_observation()
            self.generate_reward()
            self.generate_done()

    def action_inference(self, action=None, **kwargs):
        if self.action_inference_model is not None:
            self.inference_mode()
            # infer the approx. posterior on the action
            self.action_variable.init_approx_post()
            state = self.state_variable.sample()
            if self._prev_action is not None:
                action = self._prev_action
            else:
                action = self.action_variable.sample()
            inf_input = self.action_inference_model(state, action)
            # inf_input = self.action_inference_model(torch.cat((state, action), dim=1))
            self.action_variable.infer(inf_input)
            # clear_gradients(self.generative_parameters())
            self.generative_mode()

    def step_state(self, **kwargs):
        # calculate the prior on the state variable
        if self.state_prior_model is not None:
            if not self.state_variable.reinitialized:
                state = self.state_variable.sample()
                if self._prev_action is not None:
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
                if self._prev_action is not None:
                    action = self._prev_action
                else:
                    action = self.action_variable.sample()
                prior_input = self.action_prior_model(state, action)
                self.action_variable.step(prior_input)

    def generate_observation(self):
        # generate the conditional likelihood for the observation
        state = self.state_variable.sample()
        likelihood_input = self.obs_likelihood_model(state)
        self.observation_variable.generate(likelihood_input)

    def generate_reward(self):
        # generate the conditional likelihood for the reward
        state = self.state_variable.sample()
        likelihood_input = self.reward_likelihood_model(state)
        self.reward_variable.generate(likelihood_input)

    def generate_done(self):
        # generate the conditional likelihood for episode being done
        state = self.state_variable.sample()
        likelihood_input = self.done_likelihood_model(state)
        self.done_variable.generate(likelihood_input)

    def estimate_value(self, reward, done, **kwargs):
        # estimate the value of the current state
        state = self.state_variable.sample()
        value = self.value_variable(self.value_model(state)) * (1 - done)
        self.values.append(value)
        return value
