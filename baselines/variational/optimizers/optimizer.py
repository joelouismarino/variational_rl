import torch
from torch import optim
from ..misc import clear_gradients, clip_gradients, norm_gradients, divide_gradients_by_value


class Optimizer(object):
    """
    An optimizer object to handle updating the model parameters.
    """
    def __init__(self, model, lr, update_inf_online=True, clip_grad=None,
                 norm_grad=None):
        self.model = model
        self.parameters = model.parameters()
        if type(lr) == float:
            lr = {k: lr for k in self.parameters}
        self.opt = {k: optim.Adam(v, lr=lr[k]) for k, v in self.parameters.items()}
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

    def apply(self):
        for model_name, params in self.parameters.items():
            if self.update_inf_online and model_name == 'state_inference_model':
                continue
            grads = [param.grad for param in params]
            if self.clip_grad is not None:
                clip_gradients(grads, self.clip_grad)
            if self.norm_grad is not None:
                norm_gradients(grads, self.norm_grad)
            self.opt[model_name].step()

    def zero_grad(self):
        for _, opt in self.opt.items():
            opt.zero_grad()
