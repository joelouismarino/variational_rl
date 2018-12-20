import numpy as np
import torch
from torch import optim
from .misc import clear_gradients


class GradientBuffer(object):
    """
    Object to store and apply the model's parameter gradients.
    """
    def __init__(self, model, lr, capacity, batch_size):
        self.model = model
        self.gen_parameters = model.generative_parameters()
        self.inf_parameters = model.inference_parameters()
        self.gen_opt = optim.Adam(self.gen_parameters, lr=lr)
        self.inf_opt = optim.Adam(self.inf_parameters, lr=lr)
        self.current_gen_grads = None
        self.current_inf_grads = None
        self.gen_grad_buffer = []
        self.inf_grad_buffer = []
        self.capacity = capacity
        self.batch_size = batch_size
        self.n_steps = 0

    def accumulate(self):
        """
        Accumulates the current gradients.
        """
        # def _check_norm(grads):
        #     mean_norm = np.mean([grad.norm() for grad in grads])
        #     print('Gradient Norm: ' + str(mean_norm))

        # helper function for gradient accumulation
        def _accumulate(params, current_grads):
            new_grads = []
            for param in params:
                if param.grad is not None:
                    new_grads.append(param.grad.cpu())
                else:
                    new_grads.append(torch.zeros(param.shape))
            if current_grads is None:
                current_grads = new_grads
            else:
                for ind, grad in enumerate(new_grads):
                    if grad is not None:
                        current_grads[ind] += grad
            # _check_norm(current_grads)
            return current_grads

        # print('N Steps: ' + str(self.n_steps))
        # add gradients to current gradients
        self.current_gen_grads = _accumulate(self.gen_parameters, self.current_gen_grads)
        self.current_inf_grads = _accumulate(self.inf_parameters, self.current_inf_grads)

        # clear the gradients
        clear_gradients(self.gen_parameters)
        clear_gradients(self.inf_parameters)
        self.n_steps += 1

    def collect(self):
        """
        Appends the parameters' current gradients to the buffer.
        """
        # apply policy gradients
        # TODO: this should be in the model
        optimality_loss = self.model.policy_loss()
        optimality_loss.backward(retain_graph=True)
        self.accumulate()

        # TODO: should we normalize the gradients by the number of steps?

        # helper function to collect gradients
        def _collect(buffer, current_grads):
            if len(buffer) >= self.capacity:
                buffer = buffer[-self.capacity+1:-1]
            buffer.append(current_grads)
            self.n_steps = 0

        # collect current gradients into the buffer and reset
        _collect(self.gen_grad_buffer, self.current_gen_grads)
        _collect(self.inf_grad_buffer, self.current_inf_grads)
        self.current_gen_grads = None
        self.current_inf_grads = None

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
                # _check_norm([param.grad for param in parameters])
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.)
                # _check_norm([param.grad for param in parameters])
                optimizer.step()
                buffer = []
            return buffer

        self.gen_grad_buffer = _update(self.gen_parameters, self.gen_grad_buffer, self.gen_opt)
        self.inf_grad_buffer = _update(self.inf_parameters, self.inf_grad_buffer, self.inf_opt)
