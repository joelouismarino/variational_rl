import random
import torch


class DataBuffer(object):
    """
    A buffer to store episodes. Each episode is a dictionary of tensors.

    Args:
        batch_size (int): number of episodes to sample for each batch
        capacity (int): the number of episodes to store in the buffer
    """
    def __init__(self, batch_size, capacity=1e3):
        self.batch_size = batch_size
        self.capacity = int(capacity)
        self.buffer = []

    def sample(self):
        """
        Samples a batch of episodes from the buffer.
        """
        # randomly sample episodes
        indices = [random.randint(0, len(self)-1) for _ in range(self.batch_size)]
        episodes = [self.buffer[i] for i in indices]
        # for each variable, create a single tensor for all episodes
        max_len = max(episode['observation'].shape[0] for episode in episodes)
        batch = {k: torch.zeros([max_len, self.batch_size] + list(v.shape[1:])) for k, v in episodes[0].items()}
        batch['valid'] = torch.zeros(max_len, self.batch_size, 1)
        for batch_ind, episode in enumerate(episodes):
            episode_len = len(episode[list(episode.keys())[0]])
            for k in episode:
                batch[k][:episode_len, batch_ind] = episode[k]
            batch['valid'][:episode_len, batch_ind] = torch.ones(episode_len, 1)
        # self.buffer = []
        return batch

    def append(self, episode):
        """
        Removes excess episodes if capacity has been reached. Appends a new
        episode to the buffer.
        """
        if 'reconstruction' in episode:
            del episode['reconstruction']
        if 'prediction' in episode:
            del episode['prediction']
        if len(self.buffer) >= self.capacity:
            self.buffer = self.buffer[-self.capacity+1:-1]
        self.buffer.append(episode)

    def __len__(self):
        return len(self.buffer)
