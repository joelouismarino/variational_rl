import torch


class Normalizer(object):
    """
    A class to handle normalization. Based off of VecNormalize from
    common/vec_env/vec_normalize.py and common/running_mean_std.py.

    Args:
        shape (tuple): dimensions of the variable to be normalized
        shift (bool): whether to subtract mean
        scale (bool): whether to scale by std
        clip_value (float): clips to this absolute value (if greater than zero)
        epsilon (float): factor for numerical stability
    """
    def __init__(self, shape=(1), shift=True, scale=True, clip_value=0, epsilon=1e-8):
        self.rms = RunningMeanStd(shape=shape)
        self.shift = shift
        self.scale = scale
        self.clip_value = clip_value
        self.epsilon = epsilon

    def __call__(self, input, update=False):
        # normalizes the inputs, updates the running mean and std. if update is True
        # TODO: need to handle the batch dimension properly
        if update:
            self.update(input)
        if self.shift:
            input = input - self.rms.mean
        if self.scale:
            input = input / torch.sqrt(self.rms.var + self.epsilon)
        if self.clip_value > 0:
            input = torch.clamp(input, min=-self.clip_value, max=self.clip_value)
        return input

    def update(self, input):
        self.rms.update(input)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, shape, epsilon=1e-4):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon

    def _change_device(self, x):
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.var = self.var.to(x.device)

    def update(self, x):
        self._change_device(x)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count

        self.mean += delta * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count
