import copy
import torch
import torch.nn as nn
import numpy as np
from misc.collector import Collector
from modules.models import get_model
from lib.q_value_estimators import get_q_value_estimator
from lib.inference_optimizers import get_inference_optimizer


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
    """
    def __init__(self, prior_args, approx_post_args, prior_model_args,
                 q_value_estimator_args, inference_optimizer_args, misc_args):
        super(Agent, self).__init__()

        self.prior = Distribution(prior_args)
        self.target_prior = copy.deepcopy(self.prior)
        self.approx_post = Distribution(approx_post_args)

        self.prior_model = get_model(prior_model_args)
        self.target_prior_model = copy.deepcopy(self.prior_model)

        self.q_value_estimator = get_q_value_estimator(self, q_value_estimator_args)

        self.inference_optimizer = get_inference_optimizer(self, inference_optimizer_args)

        # Lagrange multipliers for KL, location KL, and scale KL
        self.log_alphas = nn.ParameterDict({'pi': nn.Parameter(torch.zeros(1)),
                                            'loc': nn.Parameter(torch.zeros(1)),
                                            'scale': nn.Parameter(torch.zeros(1))})

        # miscellaneous
        self.epsilons = misc_args['epsilons']
        self.n_action_samples = misc_args['n_action_samples']
        self.postprocess_action = misc_args['postprocess_action']
        self.reward_discount = misc_args['reward_discount']

        # mode (either 'train' or 'eval')
        self.mode = 'train'

        # collects relevant quantities
        self.collector = Collector(self)

        self.batch_size = 1

    def act(self, state, reward=None, done=False, action=None, valid=None, log_prob=None):
        state, reward, action, done, valid, log_prob = self._change_device(state, reward, action, done, valid, log_prob)
        self.generate_prior(state)
        self.inference(state)
        if self.mode == 'eval':
            if state is not None and action is None:
                action = self.approx_post.sample().detach()
        self.collector.collect(state, action, reward, done, valid, log_prob)
        action = action.tanh() if self.postprocess_action else action
        return action.cpu().numpy()

    def generate_prior(self, state):
        # generates the action prior
        if self.prior_model is not None:
            self.prior.step(self.prior_model(state=state))
            self.target_prior.step(self.target_prior_model(state=state))

    def inference(self, state):
        # infers the action approximate posterior
        self.approx_post.init(self.prior)
        self.inference_optimizer(state)

    def estimate_objective(self, state, action):
        # estimates the objective (value)
        kl = kl_divergence(self.approx_post, self.prior, samples=action)
        cond_log_like = self.q_value_estimator(state, action)
        return cond_log_like - self.alphas['pi'] * kl

    # def estimate_q_values(self, done, state, action, **kwargs):
    #     # estimate the value of the current state
    #     if action is not None:
    #         # estimate Q values for off-policy actions sample from the buffer
    #         q_value_input = [model(state=state, action=action) for model in self.q_value_models]
    #         q_values = [variable(inp) for variable, inp in zip(self.q_value_variables, q_value_input)]
    #         self.collector.qvalues1.append(q_values[0])
    #         self.collector.qvalues2.append(q_values[1])
    #
    #     # get on-policy actions and log probs
    #     if self.target_action_variable is not None:
    #         new_action = self.target_action_variable.sample(self.n_action_samples)
    #     else:
    #         new_action = self.action_variable.sample(self.n_action_samples)
    #     new_action_log_prob = self.action_variable.approx_post.log_prob(new_action).sum(dim=1).mean()
    #     self.collector.new_actions.append(new_action)
    #     if self.postprocess_action:
    #         new_action = new_action.tanh()
    #     self.collector.new_action_log_probs.append(new_action_log_prob)
    #     expanded_state = state.repeat(self.n_action_samples, 1)
    #
    #     # estimate Q value for on-policy actions sampled from current policy
    #     new_q_value_models = copy.deepcopy(self.q_value_models)
    #     new_q_value_variables = copy.deepcopy(self.q_value_variables)
    #     new_q_value_input = [model(state=expanded_state, action=new_action) for model in new_q_value_models]
    #     new_q_values = [variable(inp) for variable, inp in zip(new_q_value_variables, new_q_value_input)]
    #     sample_new_q_values = torch.min(new_q_values[0], new_q_values[1])
    #     self.collector.sample_new_q_values.append(sample_new_q_values)
    #     avg_new_q_value = torch.min(new_q_values[0].view(self.n_action_samples, -1, 1).mean(dim=0), new_q_values[1].view(self.n_action_samples, -1, 1).mean(dim=0))
    #     self.collector.new_q_values.append(avg_new_q_value)
    #
    #     # estimate target Q value for on-policy actions sampled from current policy
    #     target_q_value_input = [model(state=expanded_state, action=new_action) for model in self.target_q_value_models]
    #     target_q_values = [variable(inp).view(self.n_action_samples, -1, 1).mean(dim=0) for variable, inp in zip(self.target_q_value_variables, target_q_value_input)]
    #     target_q_value = torch.min(target_q_values[0], target_q_values[1])
    #     self.collector.target_q_values.append(target_q_value)

    def evaluate(self):
        # evaluate the objective, collect various metrics for reporting
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
        if type(reward) in [float, int]:
            reward = torch.tensor(reward).to(torch.float32).view(1, 1)
        elif type(reward) == np.ndarray:
            reward = torch.from_numpy(reward.astype('float32')).view(1, 1) # hack
        if reward.device != self.device:
            reward = reward.to(self.device)
        if action is not None:
            if type(action) == np.ndarray:
                action = torch.from_numpy(action).view(1, -1)
            if action.device != self.device:
                action = action.to(self.device)
        if type(done) == bool:
            done = torch.tensor(done).to(torch.float32).view(1, 1)
        elif type(done) == np.ndarray:
            done = torch.from_numpy(done.astype('float32')).view(1, 1) # hack
        if done.device != self.device:
            done = done.to(self.device)
        if valid is None:
            valid = torch.ones(done.shape[0], 1)
        if valid.device != self.device:
            valid = valid.to(self.device)
        if log_prob is not None:
            log_prob = log_prob.to(self.device)
        return state, reward, action, done, valid, log_prob

    def reset(self, batch_size=1, prev_action=None, prev_state=None):

        self.prior.reset(batch_size)
        self.target_prior.reset(batch_size)
        self.approx_post.reset(batch_size)
        if self.prior_model is not None:
            self.prior_model.reset(batch_size)
            self.target_prior_model.reset(batch_size)
        self.q_value_estimator.reset(batch_size, prev_action, prev_state)
        self.inference_optimizer.reset()

        # reset the collector
        self.collector.reset()

        self.batch_size = batch_size
        if prev_action is not None:
            self._prev_action = prev_action.to(self.device)
        else:
            act = self.action_variable.sample()
            self._prev_action = act.new(act.shape).zero_()
        if self.observation_variable is not None:
            if prev_obs is not None:
                self._prev_obs = prev_obs.to(self.device)
            else:
                obs = self.observation_variable.sample()
                self._prev_obs = obs.new(obs.shape).zero_()

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
        self._mode = 'train'

    def eval(self, *args):
        super(Agent, self).eval(*args)
        self._mode = 'eval'

    def parameters(self):
        param_dict = {}

        if 'parameters' in dir(self.inference_optimizer):
            param_dict['inference_optimizer'] = nn.ParameterList()
            param_dict['inference_optimizer'].extend(list(self.inference_optimizer.parameters()))
            param_dict['inference_optimizer'].extend(list(self.approx_post.parameters()))

        if self.prior_model is not None:
            param_dict['prior'] = nn.ParameterList()
            param_dict['prior'].extend(list(self.prior_model.parameters()))
            param_dict['prior'].extend(list(self.prior.parameters()))
            param_dict['target_prior'] = nn.ParameterList()
            param_dict['target_prior'].extend(list(self.target_prior_model.parameters()))
            param_dict['target_prior'].extend(list(self.target_prior.parameters()))

        q_value_param_dict = self.q_value_estimator.parameters()
        for k, v in q_value_param_dict.items():
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
        return params

    def generative_parameters(self):
        params = nn.ParameterList()
        if self.prior_model is not None:
            params.extend(list(self.prior_model.parameters()))
            params.extend(list(self.prior.parameters()))
        q_value_param_dict = self.q_value_estimator.parameters()
        for _, v in q_value_param_dict.items():
            params.extend(list(v))
        return params

    def inference_mode(self):
        self.approx_post.attach()

    def generative_mode(self):
        self.approx_post.detach()

    def load(self, state_dict):
        # load the state dictionary for the agent
        for k, v in state_dict.items():
            if hasattr(self, k):
                attr = getattr(self, k)
                try:
                    attr.load_state_dict(v)
                except:
                    print('WARNING: could not load ' + k + '.')
            else:
                raise ValueError