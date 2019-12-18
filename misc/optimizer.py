import torch
from torch import optim
from misc import clear_gradients, clip_gradients, norm_gradients, divide_gradients_by_value
import copy


class Optimizer(object):
    """
    An optimizer object to handle updating the model parameters.

    Args:
        model (Agent):
        lr (float or dict):
        clip_grad (float, optional):
        norm_grad (float, optional):
        optimizer (str, optional):
        weight_decay (float, optional):
        value_tau (float, optional):
        policy_tau (float, optional):
        value_update (str, optional):
        policy_update (str, optional):
    """
    def __init__(self, model, lr, clip_grad=None, norm_grad=None,
                 optimizer='adam', weight_decay=0., value_tau=5e-3,
                 policy_tau=5e-3, value_update='soft', policy_update='hard'):
        self.model = model
        self.parameters = model.parameters()
        if type(lr) == float:
            lr = {k: lr for k in self.parameters}
        if optimizer == 'rmsprop':
            self.opt = {k: optim.RMSprop(v, lr=lr[k], alpha=0.99, eps=1e-5, weight_decay=weight_decay) for k, v in self.parameters.items()}
        elif optimizer == 'adam':
            self.opt = {k: optim.Adam(v, lr=lr[k], weight_decay=weight_decay) for k, v in self.parameters.items()}
        else:
            raise NotImplementedError
        self.clip_grad = clip_grad
        self.norm_grad = norm_grad
        self.value_tau = value_tau
        self.policy_tau = policy_tau
        self.value_update = value_update
        self.policy_update = policy_update
        self._n_steps = 0

    def step(self, model_only=False):
        # collect and apply inference parameter gradients if necessary
        if self.model.action_inference_model is not None and not model_only:
            if self.model.action_variable.approx_post.update == 'iterative' or hasattr(self.model, 'rollout_length'):
                params = self.parameters['action_inference_model']
                grads = [param.grad for param in params]
                divide_gradients_by_value(grads, self.model.batch_size)
                divide_gradients_by_value(grads, self.model.n_inf_iter['action'])
                if self.clip_grad is not None:
                    clip_gradients(grads, self.clip_grad)
                if self.norm_grad is not None:
                    norm_gradients(grads, self.norm_grad)
                self.opt['action_inference_model'].step()
                self.opt['action_inference_model'].zero_grad()

    def apply(self, model_only=False):
        grads = []
        for model_name, params in self.parameters.items():
            grads += [param.grad for param in params]
        if self.clip_grad is not None:
            clip_gradients(grads, self.clip_grad)
        if self.norm_grad is not None:
            norm_gradients(grads, self.norm_grad)

        # if self.model.action_prior_model is not None and not model_only:
        #     # exponential moving update of the action prior
        #     old_action_prior = copy.deepcopy(self.parameters['action_prior_model'])

        # if self.model.target_action_inference_model is not None and not model_only:
        #     # copy over current action prior parameters to the target model
        #     target_params = self.parameters['target_action_inference_model']
        #     current_params = self.parameters['action_inference_model']
        #     for target_param, current_param in zip(target_params, current_params):
        #         # target_param.data.copy_(current_param.data)
        #         target_param.data.copy_(self.policy_tau * current_param.data + (1. - self.policy_tau) * target_param.data)

        for model_name, opt in self.opt.items():
            if 'target' in model_name:
                # do not update the target models with gradients
                continue
            elif model_name == 'action_inference_model' and self.model.action_variable.approx_post.update == 'iterative':
                # we optimize the iterative inference model at each step
                continue
            elif model_name == 'action_inference_model' and hasattr(self.model, 'rollout_length'):
                # same goes for the model-based agent
                continue
            elif model_name not in ['obs_likelihood_model', 'reward_likelihood_model'] and model_only:
                # if we only want to train the model, then skip the others
                continue
            opt.step()

        # if self.model.action_prior_model is not None and not model_only:
        #     # exponential moving update of the action prior
        #     current_action_prior = self.parameters['action_prior_model']
        #     for old_param, current_param in zip(old_action_prior, current_action_prior):
        #         current_param.data.copy_(self.policy_tau * current_param.data + (1. - self.policy_tau) * old_param.data)

        if self.model.target_action_prior_model is not None and not model_only:
            # copy over current action prior parameters to the target model
            target_params = self.parameters['target_action_prior_model']
            current_params = self.parameters['action_prior_model']
            for target_param, current_param in zip(target_params, current_params):
                if self.policy_update == 'hard' and self._n_steps % int(1 / self.policy_tau) == 0:
                    target_param.data.copy_(current_param.data)
                else:
                    target_param.data.copy_(self.policy_tau * current_param.data + (1. - self.policy_tau) * target_param.data)

        if self.model.target_q_value_models is not None and not model_only:
            # copy over current value parameters to the target model
            target_params = self.parameters['target_q_value_models']
            current_params = self.parameters['q_value_models']
            for target_param, current_param in zip(target_params, current_params):
                if self.value_update == 'hard' and self._n_steps % int(1 / self.value_tau) == 0:
                    target_param.data.copy_(current_param.data)
                else:
                    target_param.data.copy_(self.value_tau * current_param.data + (1. - self.value_tau) * target_param.data)

        self._n_steps += 1

    def zero_grad(self):
        for _, opt in self.opt.items():
            opt.zero_grad()
