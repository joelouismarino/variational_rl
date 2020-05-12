import copy
import torch
import torch.nn as nn
import numpy as np
from misc.collector import Collector
from lib.models import get_model
from lib.distributions import Distribution
from lib.value_estimators import get_value_estimator
from lib.inference import get_inference_optimizer
from lib.distributions import kl_divergence


class Agent(nn.Module):
    """
    Variational RL Agent

    Args:
        prior_args (dict):
        approx_post_args (dict):
        prior_model_args (dict):
        q_value_estimator_args (dict):
        inference_optimizer_args (dict):
        misc_args (dict):
        direct_inference_optimizer_args (dict):
        direct_approx_post_args (dict):
        state_value_estimator_args (dict):
    """
    def __init__(self, prior_args, approx_post_args, prior_model_args,
                 q_value_estimator_args, inference_optimizer_args, misc_args,
                 direct_inference_optimizer_args, direct_approx_post_args,
                 state_value_estimator_args):
        super(Agent, self).__init__()
        # prior
        self.prior_model = get_model(prior_model_args)
        self.target_prior_model = copy.deepcopy(self.prior_model)
        if self.prior_model is not None:
            prior_args['n_input'] = self.prior_model.n_out
        else:
            prior_args['n_input'] = None
        self.prior = Distribution(**prior_args)
        self.target_prior = Distribution(**prior_args)

        # approximate posterior
        self.inference_optimizer = get_inference_optimizer(inference_optimizer_args)
        if 'inference_model' in dir(self.inference_optimizer):
            approx_post_args['n_input'] = self.inference_optimizer.inference_model.n_out
        else:
            approx_post_args['n_input'] = None
        self.approx_post = Distribution(**approx_post_args)
        self.target_inference_optimizer = self.target_approx_post = None
        if misc_args['use_target_inference_optimizer']:
            self.target_inference_optimizer = copy.deepcopy(self.inference_optimizer)
            self.target_approx_post = Distribution(**approx_post_args)

        # optional direct inference optimizer for model-based value estimation
        self.direct_inference_optimizer = self.direct_approx_post = None
        if direct_inference_optimizer_args is not None:
            self.direct_inference_optimizer = get_inference_optimizer(direct_inference_optimizer_args)
            direct_approx_post_args['n_input'] = self.direct_inference_optimizer.inference_model.n_out
            self.direct_approx_post = Distribution(**direct_approx_post_args)

        # Q-value estimator
        self.q_value_estimator = get_value_estimator('action', q_value_estimator_args)

        # state value estimator
        self.state_value_estimator = None
        if state_value_estimator_args is not None:
            self.state_value_estimator = get_value_estimator('state', state_value_estimator_args)

        # Lagrange multipliers for KL, location KL, and scale KL
        self.log_alphas = nn.ParameterDict({'pi': nn.Parameter(torch.zeros(1)),
                                            'loc': nn.Parameter(torch.zeros(1)),
                                            'scale': nn.Parameter(torch.zeros(1))})

        # miscellaneous
        self.epsilons = misc_args['epsilons']
        self.n_action_samples = misc_args['n_action_samples']
        self.n_q_action_samples = misc_args['n_q_action_samples']
        self.postprocess_action = misc_args['postprocess_action']
        self.reward_discount = misc_args['reward_discount']
        self.retrace_lambda = misc_args['retrace_lambda']
        self.model_value_targets = misc_args['model_value_targets']
        self.optimize_targets = misc_args['optimize_targets']
        self.direct_targets = misc_args['direct_targets']
        self.off_policy_targets = misc_args['off_policy_targets']

        # mode (either 'train' or 'eval')
        self.mode = 'train'

        # collects relevant quantities
        self.collector = Collector(self)

        # internal variables
        self.batch_size = 1
        self._prev_action = self._prev_state = None

    def act(self, state, reward=None, done=False, action=None, valid=None, log_prob=None, eval=False):
        """
        The main function through which the agent interacts with the environment.

        Args:
            state (torch.Tensor): state of shape [batch_size, n_state_dims]
            reward (torch.Tensor): reward of shape [batch_size, 1]
            done ():
            action (torch.Tensor): action of shape [batch_size, n_action_dims]
            valid ():
            log_prob (torch.Tensor): log probability of action [batch_size, 1]
            eval (bool): whether to estimate the MAP action
        """
        state, reward, action, done, valid, log_prob = self._change_device(state, reward, action, done, valid, log_prob)
        self.generate_prior(state)
        self.inference(state)
        if self.mode == 'eval':
            if state is not None and action is None:
                action = self.approx_post.sample(argmax=eval).detach()
        self.collector.collect(state, action, reward, done, valid, log_prob)
        self.step(state, action)
        action = action.tanh() if self.postprocess_action and self.mode == 'eval' else action
        return action.cpu().numpy()

    def generate_prior(self, state, detach_params=False):
        """
        Generates the action prior.

        Args:
            state (torch.Tensor): state of shape [batch_size, n_state_dims]
            detach_params (bool): whether to detach the prior parameters
        """
        if self.prior_model is not None:
            if detach_params:
                prior_model = copy.deepcopy(self.prior_model)
                target_prior_model = copy.deepcopy(self.target_prior_model)
            else:
                prior_model = self.prior_model
                target_prior_model = self.target_prior_model
            self.prior.step(prior_model(state=state), detach_params=detach_params)
            self.target_prior.step(target_prior_model(state=state), detach_params=detach_params)

    def inference(self, state, detach_params=False, direct=False):
        """
        Infers the action approximate posterior.

        Args:
            state (torch.Tensor): state of shape [batch_size, n_state_dims]
            detach_params (bool): whether to detach the inference optimizer parameters
            direct (bool): whether to use the direct inference optimizer
        """
        if self.direct_inference_optimizer is not None:
            self.direct_approx_post.init(self.prior)
            self.direct_inference_optimizer(self, state, detach_params=detach_params, direct=True)
        if self.inference_optimizer.n_inf_iters == 1 or not direct:
            # run the inference model if it is already direct
            self.approx_post.init(self.prior)
            self.inference_optimizer(self, state, detach_params=detach_params)
            if self.target_approx_post is not None and self.mode == 'train':
                # get the target estimate
                self.target_approx_post.init(self.prior)
                self.target_inference_optimizer(self, state, detach_params=detach_params, target=True)

    def estimate_objective(self, state, action, target=False):
        """
        Estimates the objective (state-value).

        Args:
            state (torch.Tensor): state of shape [batch_size, n_state_dims]
            actions (torch.Tensor): action of shape [n_action_samples * batch_size, n_action_dims]
            target (bool): whether to use the target approx post

        Returns objective estimate of shape [n_action_samples * batch_size, 1]
        """
        approx_post = self.target_approx_post if target else self.approx_post
        kl = kl_divergence(approx_post, self.prior, n_samples=self.n_action_samples, sample=action).sum(dim=1, keepdim=True)
        expanded_state = state.repeat(self.n_action_samples, 1)
        cond_log_like = self.q_value_estimator(self, expanded_state, action, detach_params=True, target=self.optimize_targets)
        return cond_log_like - self.alphas['pi'] * kl.repeat(self.n_action_samples, 1)

    def step(self, state, action):
        """
        Updates the previous state and action (for state and reward prediction).

        Args:
            state (torch.Tensor): state of shape [batch_size, n_state_dims]
            action (torch.Tensor): action of shape [batch_size, n_action_dims]
        """
        action = action.tanh() if self.postprocess_action and self.mode == 'eval' else action
        self._prev_action = action; self._prev_state = state
        self.q_value_estimator.set_prev_state(state)

    def evaluate(self):
        """
        Evaluate the objective, collect various metrics for reporting.
        """
        objective = self.collector.evaluate()
        objective.backward()
        results = {}
        for k, v in self.collector.get_metrics().items():
            results[k] = v
        for k, v in self.collector.get_inf_imp().items():
            results[k] = v
        for k, v in self.collector.get_grads().items():
            results[k] = v
        return results

    def _change_device(self, state, reward, action, done, valid, log_prob):
        if state is None:
            state = torch.zeros(self.collector.episode['state'][0].shape)
        elif type(state) == np.ndarray:
            state = torch.from_numpy(state.astype('float32')).view(1, -1) # hack
        if state.device != self.device:
            state = state.to(self.device)
        if reward is not None:
            if type(reward) in [float, int]:
                reward = torch.tensor(reward).to(torch.float32).view(1, 1)
            elif type(reward) == np.ndarray:
                reward = torch.from_numpy(reward.astype('float32')).view(1, 1) # hack
            if reward.device != self.device:
                reward = reward.to(self.device)
        else:
            reward = torch.tensor(0.).to(torch.float32).view(1, 1)
        if action is not None:
            if type(action) == np.ndarray:
                action = torch.from_numpy(action).view(1, -1)
            if action.device != self.device:
                action = action.to(self.device)
        if done is not None:
            if type(done) == bool:
                done = torch.tensor(done).to(torch.float32).view(1, 1)
            elif type(done) == np.ndarray:
                done = torch.from_numpy(done.astype('float32')).view(1, 1) # hack
            if done.device != self.device:
                done = done.to(self.device)
        else:
            done = torch.tensor(False).to(torch.float32).view(1, 1)
        if valid is None:
            valid = torch.ones(done.shape[0], 1)
        if valid.device != self.device:
            valid = valid.to(self.device)
        if log_prob is not None:
            log_prob = log_prob.to(self.device)
        return state, reward, action, done, valid, log_prob

    def reset(self, batch_size=1, prev_action=None, prev_state=None):
        """
        Reset the agent.

        Args:
            batch_size (int): the batch size of the data
            prev_action (torch.Tensor): previous actions (used for training model)
            prev_state (torch.Tensor): previous states (used for training model)
        """
        if prev_action is not None:
            prev_action = prev_action.to(self.device)
            prev_state = prev_state.to(self.device)
        self.prior.reset(batch_size)
        self.target_prior.reset(batch_size)
        self.approx_post.reset(batch_size)
        if self.target_approx_post is not None:
            self.target_approx_post.reset(batch_size)
        if self.prior_model is not None:
            self.prior_model.reset(batch_size)
            self.target_prior_model.reset(batch_size)
        self.q_value_estimator.reset(batch_size, prev_state)
        if self.state_value_estimator is not None:
            self.state_value_estimator.reset(batch_size)
        self.inference_optimizer.reset(batch_size)
        if self.direct_approx_post is not None:
            self.direct_approx_post.reset(batch_size)
            self.direct_inference_optimizer.reset(batch_size)

        # reset the collector
        self.collector.reset()

        self.batch_size = batch_size
        self._prev_action = prev_action
        self._prev_state = prev_state

        # clamp log-alphas to prevent collapse
        for name, log_alpha in self.log_alphas.items():
            log_alpha = torch.clamp(log_alpha, min=-15.)

    @property
    def device(self):
        p = self.parameters()
        return p[list(p.keys())[0]][0].device

    @property
    def alphas(self):
        return {name: self.log_alphas[name].exp().detach() for name in self.log_alphas}

    def train(self, *args):
        super(Agent, self).train(*args)
        self.mode = 'train'

    def eval(self, *args):
        super(Agent, self).eval(*args)
        self.mode = 'eval'

    def parameters(self):
        param_dict = {}

        if 'parameters' in dir(self.inference_optimizer):
            param_dict['inference_optimizer'] = nn.ParameterList()
            param_dict['inference_optimizer'].extend(list(self.inference_optimizer.parameters()))
            param_dict['inference_optimizer'].extend(list(self.approx_post.parameters()))
            if self.target_approx_post is not None:
                param_dict['target_inference_optimizer'] = nn.ParameterList()
                param_dict['target_inference_optimizer'].extend(list(self.target_inference_optimizer.parameters()))
                param_dict['target_inference_optimizer'].extend(list(self.target_approx_post.parameters()))

        if self.prior_model is not None:
            param_dict['prior'] = nn.ParameterList()
            param_dict['prior'].extend(list(self.prior_model.parameters()))
            param_dict['prior'].extend(list(self.prior.parameters()))
            param_dict['target_prior'] = nn.ParameterList()
            param_dict['target_prior'].extend(list(self.target_prior_model.parameters()))
            param_dict['target_prior'].extend(list(self.target_prior.parameters()))

        if self.direct_approx_post is not None:
            param_dict['direct_inference_optimizer'] = nn.ParameterList()
            param_dict['direct_inference_optimizer'].extend(list(self.direct_inference_optimizer.parameters()))
            param_dict['direct_inference_optimizer'].extend(list(self.direct_approx_post.parameters()))

        q_value_param_dict = self.q_value_estimator.parameters()
        for k, v in q_value_param_dict.items():
            param_dict[k] = v

        if self.state_value_estimator is not None:
            value_param_dict = self.state_value_estimator.parameters()
            for k, v in value_param_dict.items():
                param_dict[k] = v

        if self.log_alphas is not None:
            param_dict['log_alphas'] = nn.ParameterList()
            for name in self.log_alphas:
                param_dict['log_alphas'].append(self.log_alphas[name])

        return param_dict

    def inference_parameters(self):
        params = nn.ParameterList()
        if 'parameters' in dir(self.inference_optimizer):
            params.extend(list(self.inference_optimizer.parameters()))
            params.extend(list(self.approx_post.parameters()))
        if self.direct_approx_post is not None:
            params.extend(list(self.direct_inference_optimizer.parameters()))
            params.extend(list(self.direct_approx_post.parameters()))
        return params

    def generative_parameters(self):
        params = nn.ParameterList()
        if self.prior_model is not None:
            params.extend(list(self.prior_model.parameters()))
            params.extend(list(self.prior.parameters()))
        q_value_param_dict = self.q_value_estimator.parameters()
        for _, v in q_value_param_dict.items():
            params.extend(list(v))
        if self.state_value_estimator is not None:
            value_param_dict = self.state_value_estimator.parameters()
            for _, v in value_param_dict.items():
                params.extend(list(v))
        return params
