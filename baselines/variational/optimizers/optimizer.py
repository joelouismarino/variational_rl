import torch
from torch import optim
from ..misc import clear_gradients, clip_gradients, norm_gradients, divide_gradients_by_value


class Optimizer(object):
    """
    An optimizer object to handle updating the model parameters.
    """
    def __init__(self, model, lr, update_inf_online=True, clip_grad=None,
                 norm_grad=None, optimizer='rmsprop'):
        self.model = model
        self.parameters = model.parameters()
        if type(lr) == float:
            lr = {k: lr for k in self.parameters}
        if optimizer == 'rmsprop':
            self.opt = {k: optim.RMSprop(v, lr=lr[k], alpha=0.99, eps=1e-5) for k, v in self.parameters.items()}
        elif optimizer == 'adam':
            self.opt = {k: optim.Adam(v, lr=lr[k]) for k, v in self.parameters.items()}
        else:
            raise NotImplementedError
        self.update_inf_online = update_inf_online
        self.clip_grad = clip_grad
        self.norm_grad = norm_grad

    def step(self):
        # collect and apply inference parameter gradients if necessary
        if self.update_inf_online:
            if self.model.state_inference_model is not None:
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
            if self.update_inf_online and model_name == 'state_inference_model':
                continue
            if self.update_inf_online and model_name == 'action_inference_model':
                continue
            grads += [param.grad for param in params]
        if self.clip_grad is not None:
            clip_gradients(grads, self.clip_grad)
        if self.norm_grad is not None:
            norm_gradients(grads, self.norm_grad)
        for _, opt in self.opt.items():
            opt.step()

    def zero_grad(self):
        for _, opt in self.opt.items():
            opt.zero_grad()
