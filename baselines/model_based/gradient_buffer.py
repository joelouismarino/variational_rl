import numpy as np
import torch
from torch import optim
from .misc import clear_gradients, clip_gradients, norm_gradients, normalize_gradients_by_time


class GradientBuffer(object):
    """
    Object to store and apply the model's parameter gradients.

    Args:
        model (Model): the model to optimize
        lr (float or dict): learning rate(s)
        capacity (int): the size of the gradient buffer
        batch_size (int): the number of gradients to average per batch
        clip_grad (float): clips gradients to the (absolute) value clip_grad
        norm_grad (float): normalizes gradients to have a norm of norm_grad
    """
    def __init__(self, model, lr, capacity, batch_size, clip_grad=None, norm_grad=None):
        self.model = model
        self.parameters = model.parameters()
        if type(lr) == float:
            lr = {k: lr for k in self.parameters}
        self.opt = {k: optim.Adam(v, lr=lr[k]) for k, v in self.parameters.items()}
        # self.current_grads = {k: None for k in self.parameters}
        self.grad_buffer = {k: [] for k in self.parameters}
        self.capacity = capacity
        self.batch_size = batch_size
        self.clip_grad = clip_grad
        self.norm_grad = norm_grad
        self.n_steps = 0

    # def accumulate(self):
    #     """
    #     Accumulates the current gradients.
    #     """
    #     # def _check_norm(grads):
    #     #     mean_norm = np.mean([grad.norm() for grad in grads])
    #     #     print('Gradient Norm: ' + str(mean_norm))
    #
    #     # helper function for gradient accumulation
    #     def _accumulate(params, current_grads):
    #         new_grads = []
    #         for param in params:
    #             if param.grad is not None:
    #                 new_grads.append(param.grad.cpu())
    #             else:
    #                 new_grads.append(torch.zeros(param.shape))
    #         if current_grads is None:
    #             current_grads = new_grads
    #         else:
    #             for ind, grad in enumerate(new_grads):
    #                 if grad is not None:
    #                     current_grads[ind] += grad
    #         # _check_norm(current_grads)
    #         return current_grads
    #
    #     # add gradients to current gradients
    #     for model_name in self.parameters:
    #         self.current_grads[model_name] = _accumulate(self.parameters[model_name], self.current_grads[model_name])
    #         # clear the gradients
    #         clear_gradients(self.parameters[model_name])
    #
    #     self.n_steps += 1

    def evaluate(self):
        """
        Evaluates the free energy and sets gradients using backprop and REINFORCE.
        """
        self.n_steps = len(self.model.objectives['observation']) - 1
        # compute the free energy at each time step of the episode
        free_energy = torch.zeros(self.n_steps+1)
        for objective_name, objective in self.model.objectives.items():
            free_energy = free_energy + torch.stack(objective)

        # calculate sums of future objective terms for REINFORCE gradients
        future_sums = torch.flip(torch.cumsum(torch.flip(free_energy.detach(), dims=[0]), dim=0), dims=[0])
        if future_sums.shape[0] > 2:
            future_sums = (future_sums - future_sums[1:].mean()) / (future_sums[1:].std() + 1e-6)
        log_probs = torch.stack(self.model.log_probs['action'])
        reinforce_terms = - log_probs * future_sums

        modified_free_energy = free_energy + reinforce_terms
        modified_free_energy.sum().backward()

    def collect(self):
        """
        Appends the parameters' gradients to the buffer. Clears current gradients.
        """
        def _collect(params, buffer):
            # collect the new gradients from the parameters
            new_grads = []
            for param in params:
                if param.grad is not None:
                    new_grads.append(param.grad.cpu())
                else:
                    new_grads.append(torch.zeros(param.shape))
            # clear the gradients
            clear_gradients(params)

            # normalize and clip the gradients
            normalize_gradients_by_time(new_grads, self.n_steps)
            if self.clip_grad is not None:
                clip_gradients(new_grads, self.clip_grad)
            if self.norm_grad is not None:
                norm_gradients(new_grads, self.norm_grad)

            # append the new gradients to the gradient buffer
            if len(buffer) >= self.capacity:
                buffer = buffer[-self.capacity+1:-1]
            buffer.append(new_grads)
            return new_grads

        # collect new gradients into the buffer and reset
        episode_grads = {}
        for model_name in self.parameters:
            episode_grads[model_name] = _collect(self.parameters[model_name], self.grad_buffer[model_name])

        self.n_steps = 0
        return episode_grads


    # def collect(self):
    #     """
    #     Appends the parameters' current gradients to the buffer.
    #     """
    #
    #     def _normalize_gradients_by_time(grads, steps):
    #         for g in filter(lambda g: g is not None, grads):
    #             g.data.div_(steps)
    #
    #     # helper function to collect gradients
    #     def _collect(buffer, current_grads):
    #         if len(buffer) >= self.capacity:
    #             buffer = buffer[-self.capacity+1:-1]
    #         _normalize_gradients_by_time(current_grads, self.n_steps)
    #         if self.clip_grad is not None:
    #             clip_gradients(current_grads, self.clip_grad)
    #         if self.norm_grad is not None:
    #             norm_gradients(current_grads, self.norm_grad)
    #         buffer.append(current_grads)
    #         return current_grads
    #
    #     # collect current gradients into the buffer and reset
    #     episode_grads = {}
    #     for model_name in self.parameters:
    #         episode_grads[model_name] = _collect(self.grad_buffer[model_name], self.current_grads[model_name])
    #         self.current_grads[model_name] = None
    #
    #     self.n_steps = 0
    #     return episode_grads

    def update(self):
        """
        Updates the parameters using gradients from the buffer.
        """
        # TODO: update based on gradient variance instead of fixed batch size,
        #       also, keep gradients but randomly sample?
        #       also, is there a better way to store/average gradients?

        # def _check_norm(grads):
        #     mean_norm = np.mean([grad.norm() for grad in grads])
        #     print('Gradient Norm: ' + str(mean_norm))

        def _update(parameters, buffer, optimizer):
            if len(buffer) >= self.batch_size:
                mean_grads = [torch.zeros(param.shape) for param in parameters]
                for i in range(len(parameters)):
                    for j in range(self.batch_size):
                        mean_grads[i] += buffer[j][i]
                    mean_grads[i] /= self.batch_size
                    parameters[i].grad = mean_grads[i]
                optimizer.step()
                buffer = []
            return buffer

        for model_name in self.parameters:
            self.grad_buffer[model_name] = _update(self.parameters[model_name],
                                                   self.grad_buffer[model_name],
                                                   self.opt[model_name])
