import torch
from torch import optim
from misc import clear_gradients, clip_gradients, norm_gradients, divide_gradients_by_value
import copy


class Optimizer(object):
    """
    An optimizer object to handle updating the model parameters.

    Args:
        agent (Agent):
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
    def __init__(self, agent, lr, clip_grad=None, norm_grad=None,
                 optimizer='adam', weight_decay=0., value_tau=5e-3,
                 policy_tau=5e-3, value_update='soft', policy_update='hard'):
        self.agent = agent
        self.parameters = agent.parameters()
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

    def apply(self, model_only=False, critic_only=False, actor_only=False):

        # divide optimizer gradients by number of inference iterations and batch size
        if 'inference_optimizer' in self.parameters.keys():
            params = self.parameters['inference_optimizer']
            grads = [param.grad for param in params]
            divide_gradients_by_value(grads, self.agent.inference_optimizer.n_inf_iters)
            divide_gradients_by_value(grads, self.agent.batch_size)

        grads = []
        for model_name, params in self.parameters.items():
            grads += [param.grad for param in params]
        if self.clip_grad is not None:
            clip_gradients(grads, self.clip_grad)
        if self.norm_grad is not None:
            norm_gradients(grads, self.norm_grad)

        for model_name, opt in self.opt.items():
            if 'target' in model_name:
                # do not update the target models with gradients
                continue
            elif model_name not in ['state_likelihood_model', 'reward_likelihood_model'] and model_only:
                continue
            elif model_name not in ['q_value_models'] and critic_only:
                continue
            elif model_name not in ['inference_optimizer', 'prior'] and actor_only:
                continue
            opt.step()

        if self.agent.target_prior_model is not None and not model_only:
            # copy over current action prior parameters to the target model
            target_params = self.parameters['target_prior']
            current_params = self.parameters['prior']
            for target_param, current_param in zip(target_params, current_params):
                if self.policy_update == 'hard' and self._n_steps % int(1 / self.policy_tau) == 0:
                    target_param.data.copy_(current_param.data)
                else:
                    target_param.data.copy_(self.policy_tau * current_param.data + (1. - self.policy_tau) * target_param.data)

        # if self.agent.q_value_estimator.target_q_value_models is not None and not model_only:
        if self.agent.q_value_estimator.target_q_value_models is not None:
            # copy over current value parameters to the target model
            target_params = self.parameters['target_q_value_models']
            current_params = self.parameters['q_value_models']
            for target_param, current_param in zip(target_params, current_params):
                if self.value_update == 'hard' and self._n_steps % int(1 / self.value_tau) == 0:
                    target_param.data.copy_(current_param.data)
                else:
                    target_param.data.copy_(self.value_tau * current_param.data + (1. - self.value_tau) * target_param.data)

        # TODO: only update this when updating the critic?
        self._n_steps += 1

    def zero_grad(self):
        for _, opt in self.opt.items():
            opt.zero_grad()
