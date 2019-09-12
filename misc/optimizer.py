import torch
from torch import optim
from misc import clear_gradients, clip_gradients, norm_gradients, divide_gradients_by_value


class Optimizer(object):
    """
    An optimizer object to handle updating the model parameters.
    """
    def __init__(self, model, lr, clip_grad=None, norm_grad=None,
                 optimizer='adam', weight_decay=0., ema_tau=5e-3):
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
        self.ema_tau = ema_tau

    def step(self):
        # collect and apply inference parameter gradients if necessary
        if self.model.state_inference_model is not None:
            if self.model.state_variable.approx_post.update == 'iterative':
                params = self.parameters['state_inference_model']
                grads = [param.grad for param in params]
                divide_gradients_by_value(grads, self.model.batch_size)
                divide_gradients_by_value(grads, self.model.n_inf_iter['state'])
                if self.clip_grad is not None:
                    clip_gradients(grads, self.clip_grad)
                if self.norm_grad is not None:
                    norm_gradients(grads, self.norm_grad)
                self.opt['state_inference_model'].step()
                self.opt['state_inference_model'].zero_grad()

        if self.model.action_inference_model is not None:
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

    def apply(self):
        grads = []
        for model_name, params in self.parameters.items():
            grads += [param.grad for param in params]
        if self.clip_grad is not None:
            clip_gradients(grads, self.clip_grad)
        if self.norm_grad is not None:
            norm_gradients(grads, self.norm_grad)
        for model_name, opt in self.opt.items():
            if 'target' in model_name:
                # do not update the target value models
                continue
            elif model_name == 'action_inference_model' and self.model.action_variable.approx_post.update == 'iterative':
                # we optimize the iterative inference model at each step
                continue
            elif model_name == 'action_inference_model' and hasattr(self.model, 'rollout_length'):
                # same goes for the model-based agent
                continue
            opt.step()

        if self.model.target_q_value_models is not None:
            # exponential moving update of the target q value model parameters
            target_params = self.parameters['target_q_value_models']
            current_params = self.parameters['q_value_models']
            for target_param, current_param in zip(target_params, current_params):
                target_param.data.copy_(self.ema_tau * current_param.data + (1. - self.ema_tau) * target_param.data)

    def zero_grad(self):
        for _, opt in self.opt.items():
            opt.zero_grad()
