import random
import torch
import copy

# used so that we sample the ends of sequences more often
DELTA = 0

class Buffer(object):
    """
    A buffer to store episodes. Each episode is a dictionary of tensors.

    Args:
        batch_size (int): number of trajectories to sample for each batch
        sequence_length (int): length of sequences to sample, use 0 for entire episodes
        capacity (int): the number of time steps to store in the buffer
    """
    def __init__(self, batch_size, seq_len=0, capacity=1e6):
        self.batch_size = batch_size
        self.capacity = int(capacity)
        self.sequence_length = int(seq_len)
        self.buffer = []
        self.total_steps = 0
        self.current_steps = 0
        self.last_episode = None

    def sample(self):
        """
        Samples a batch of episodes from the buffer.
        """
        # randomly sample episodes
        indices = [random.randint(0, len(self.buffer)-1) for _ in range(self.batch_size)]
        episodes = [self.buffer[i] for i in indices]
        # for each variable, create a single tensor for all episodes
        seq_len = self.sequence_length
        if seq_len <= 0:
            # use entire episodes
            seq_len = max(episode['state'].shape[0] for episode in episodes)
        batch = {k: torch.zeros([seq_len, self.batch_size] + list(v.shape[1:])) for k, v in episodes[0].items()}
        batch['valid'] = torch.zeros(seq_len, self.batch_size, 1)
        batch['prev_state'] = torch.zeros([self.batch_size] + list(episodes[0]['state'].shape[1:]))
        batch['prev_action'] = torch.zeros([self.batch_size] + list(episodes[0]['action'].shape[1:]))
        for batch_ind, episode in enumerate(episodes):
            episode_len = len(episode[list(episode.keys())[0]])
            start_ind = 0
            end_ind = episode_len
            if seq_len < episode_len:
                # select a sub-sequence of the episode
                start_ind = random.randint(0, episode_len - seq_len + DELTA)
                if start_ind > episode_len:
                    start_ind = episode_len - 1
                end_ind = min(start_ind + seq_len, end_ind)
            l = end_ind - start_ind
            for k in episode:
                batch[k][:l, batch_ind] = episode[k][start_ind:end_ind]
            batch['valid'][:l, batch_ind] = torch.ones(l, 1)
            batch['prev_state'][batch_ind] = episode['state'][start_ind-1]
            batch['prev_action'][batch_ind] = episode['action'][start_ind-1]
        # self.buffer = []
        return batch

    def append(self, episode):
        """
        Removes excess episodes if capacity has been reached. Appends a new
        episode to the buffer.
        """
        self.last_episode = copy.deepcopy(episode)
        keys_to_remove = []
        for k, v in episode.items():
            if type(v) == dict:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del episode[k]
        if 'value' in episode:
            del episode['value']
        if 'advantage' in episode:
            del episode['advantage']
        if 'return' in episode:
            del episode['return']
        self.total_steps += episode['state'].shape[0]
        self.current_steps += episode['state'].shape[0]
        # pop any old episodes from the buffer
        while self.current_steps > self.capacity:
            popped_episode = self.buffer[0]
            self.buffer = self.buffer[1:]
            self.current_steps -= popped_episode['state'].shape[0]
        self.buffer.append(episode)

    def empty(self):
        """
        Emptys the buffer.
        """
        self.buffer = []
        self.total_steps = 0
        self.current_steps = 0
        self.last_episode = None

    def __len__(self):
        """
        Number of episodes.
        """
        return self.current_steps
