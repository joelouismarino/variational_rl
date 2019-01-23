# import torch
from .config import get_model_args
from .models import Model
from .buffers import DataBuffer
from .optimizers import Optimizer
from .util.plot_util import Plotter
from .util.log_util import Logger
from .util.print_util import print_step_metrics, print_episode_metrics
from .util.train_util import collect_episode, train


def learn(env, seed, total_timesteps, log_dir, batch_size=15, n_updates=3,
          n_initial_batches=1, lr=0.001, device=None, **kwargs):

    # torch.manual_seed(seed)

    logger = Logger(log_dir)
    plotter = Plotter(logger.log_str)

    # create the model
    model_args = get_model_args(env)
    model = Model(**model_args)
    if device is not None:
        model.to(device)
    model.reset()

    # create the buffer
    buffer = DataBuffer(batch_size=batch_size)

    # create the optimizer
    base_lr = lr
    lr = {'state_inference_model': base_lr/10,
          'action_inference_model': base_lr/10,
          'state_prior_model': base_lr,
          'action_prior_model': 0.,
          'obs_likelihood_model': base_lr,
          'reward_likelihood_model': base_lr,
          'done_likelihood_model': base_lr}
    optimizer = Optimizer(model, lr=lr)

    # collect episodes and train
    timestep = 0
    n_episodes = 0
    while timestep < total_timesteps:

        # collect an episode
        print('Collecting Episode: ' + str(n_episodes + 1))
        episode, episode_length = collect_episode(env, model)
        timestep += episode_length
        n_episodes += 1
        plotter.plot_episode(episode)
        # logger.log_episode(episode)
        buffer.append(episode)

        # train on samples from buffer
        if len(buffer) >= n_initial_batches * batch_size:
            print('Training Model.')
            for update in range(n_updates):
                print(' Batch: ' + str(update + 1))
                batch = buffer.sample()
                results = train(model, batch, optimizer)
                logger.log_train_step(results)
                plotter.plot_train_step(results)

    return model
