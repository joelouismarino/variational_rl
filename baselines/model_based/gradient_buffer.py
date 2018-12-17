from torch import optim


class GradientBuffer(object):
    """
    Object to store and apply the model's parameter gradients.
    """
    def __init__(self, model, lr, capacity, batch_size):
        self.gen_parameters = model.generative_parameters()
        self.inf_parameters = model.inference_parameters()
        self.gen_opt = optim.Adam(self.gen_parameters, lr=lr)
        self.inf_opt = optim.Adam(self.inf_parameters, lr=lr)
        self.gen_grad_buffer = []
        self.inf_grad_buffer = []
        self.capacity = capacity
        self.batch_size = batch_size

    def collect(self):
        """
        Appends the parameters' current gradients to the buffer.
        """
        if len(self.gen_grad_buffer) >= self.capacity:
            self.gen_grad_buffer = self.gen_grad_buffer[-self.capacity+1:-1]
        gen_grads = [param.grad.cpu() for param in self.gen_parameters]
        self.gen_grad_buffer.append(gen_grads)

        if len(self.inf_grad_buffer) >= self.capacity:
            self.inf_grad_buffer = self.inf_grad_buffer[-self.capacity+1:-1]
        inf_grads = [param.grad.cpu() for param in self.inf_parameters]
        self.inf_grad_buffer.append(inf_grads)

    def update(self):
        """
        Updates the parameters using gradients from the buffer.
        """
        # TODO: update based on gradient variance instead of fixed batch size,
        #       also, keep gradients but randomly sample?
        #       also, is there a better way to store/average gradients?

        if len(self.gen_grad_buffer) > self.batch_size:
            mean_grads = [0 for _ in range(len(self.gen_parameters))]
            for i in range(len(self.gen_parameters)):
                for j in range(self.batch_size):
                    mean_grads[i] += self.gen_grad_buffer[j][i]
                mean_grads[i] /= self.batch_size
                self.gen_parameters[i].grad = mean_grads[i]
            self.gen_opt.step()
            self.gen_grad_buffer = []

        if len(self.inf_grad_buffer) > self.batch_size:
            mean_grads = [0 for _ in range(len(self.inf_parameters))]
            for i in range(len(self.inf_parameters)):
                for j in range(self.batch_size):
                    mean_grads[i] += self.inf_grad_buffer[j][i]
                mean_grads[i] /= self.batch_size
                self.inf_parameters[i].grad = mean_grads[i]
            self.inf_opt.step()
            self.inf_grad_buffer = []
