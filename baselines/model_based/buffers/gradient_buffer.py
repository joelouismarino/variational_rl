import torch
from torch import optim
from ..misc import clear_gradients, clip_gradients, norm_gradients, normalize_gradients_by_time


class GradientBuffer(object):
    """
    Object to store and apply the model's parameter gradients.

    Args:
        model (Model): the model to optimize
        lr (float or dict): learning rate(s)
        capacity (int): the size of the gradient buffer
        batch_size (int): the number of gradients to average per batch
        update_inf_online (bool): whether to update inference parameters online
        clip_grad (float): clips gradients to the (absolute) value clip_grad
        norm_grad (float): normalizes gradients to have a norm of norm_grad
    """
    def __init__(self, model, lr, capacity, batch_size, update_inf_online=False,
                 clip_grad=None, norm_grad=None):
        self.model = model
        self.parameters = model.parameters()
        if type(lr) == float:
            lr = {k: lr for k in self.parameters}
        self.opt = {k: optim.Adam(v, lr=lr[k]) for k, v in self.parameters.items()}
        self.grad_buffer = {k: [] for k in self.parameters}
        self.capacity = capacity
        self.batch_size = batch_size
        self.update_inf_online = update_inf_online
        self.clip_grad = clip_grad
        self.norm_grad = norm_grad
        self.n_steps = 0

    def evaluate(self):
        """
        Evaluates the free energy and sets gradients using backprop and REINFORCE.
        """
        self.n_steps = len(self.model.objectives['observation']) - 1
        # compute the free energy at each time step of the episode
        free_energy = torch.zeros(self.n_steps+1).to(self.model.device)
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

    def step(self):
        """
        Updates inference parameters (if update_inf_online == True).
        """
        if self.update_inf_online:
            for model_name in self.parameters:
                if model_name == 'state_inference_model':
                    # collect the gradients
                    self._collect(self.parameters[model_name],
                                  self.grad_buffer[model_name], 1)
                    # update the parameters
                    self.grad_buffer[model_name] = self._update(self.parameters[model_name],
                                                                self.grad_buffer[model_name],
                                                                self.opt[model_name])

    def _collect(self, params, buffer, n_steps):
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
        normalize_gradients_by_time(new_grads, n_steps)
        if self.clip_grad is not None:
            clip_gradients(new_grads, self.clip_grad)
        if self.norm_grad is not None:
            norm_gradients(new_grads, self.norm_grad)

        # append the new gradients to the gradient buffer
        if len(buffer) >= self.capacity:
            buffer = buffer[-self.capacity+1:-1]
        buffer.append(new_grads)
        return new_grads

    def collect(self):
        """
        Appends the parameters' gradients to the buffer. Clears current gradients.
        """
        # collect new gradients into the buffer and reset
        episode_grads = {}
        for model_name in self.parameters:
            episode_grads[model_name] = self._collect(self.parameters[model_name], self.grad_buffer[model_name], self.n_steps)
        self.n_steps = 0
        return episode_grads

    def _update(self, parameters, buffer, optimizer):
        # TODO: update based on gradient variance instead of fixed batch size,
        #       also, keep gradients but randomly sample?
        #       also, is there a better way to store/average gradients?

        # TODO: should be sampling gradients from the buffer. Currently just
        #       taking the most recent batch.
        if len(buffer) >= self.batch_size:
            mean_grads = [torch.zeros(param.shape) for param in parameters]
            for i in range(len(parameters)):
                for j in range(self.batch_size):
                    mean_grads[i] += buffer[j][i]
                mean_grads[i] /= self.batch_size
                parameters[i].grad = mean_grads[i].to(self.model.device)
            optimizer.step()
            buffer = []
        return buffer

    def update(self):
        """
        Updates the parameters using gradients from the buffer.
        """
        for model_name in self.parameters:
            self.grad_buffer[model_name] = self._update(self.parameters[model_name],
                                                        self.grad_buffer[model_name],
                                                        self.opt[model_name])
