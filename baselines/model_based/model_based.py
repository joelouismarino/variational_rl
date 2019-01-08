# import torch
from .config import get_model_args
from .models import Model
from .gradient_buffer import GradientBuffer
from .util import Viewer
from .util.plot_util import Plotter
from .util.log_util import Logger
from .util.print_util import print_step_metrics, print_episode_metrics


def learn(env, seed, total_timesteps, log_dir, **kwargs):

    # torch.manual_seed(seed)

    logger = Logger(log_dir)
    plotter = Plotter(logger.log_str)

    model_args = get_model_args(env)
    model = Model(**model_args)
    model.training = True

    grad_buffer = GradientBuffer(model, lr=0.001, capacity=100, batch_size=5)

    observation = env.reset()
    reward = None
    episode = 0

    for step in range(total_timesteps):

        action = model.act(observation, reward)
        logger.log_step(model, observation, reward)
        observation, reward, done, _ = env.step(action)
        grad_buffer.accumulate()

        if step % 25 == 0:
            plotter.plot_image(observation, 'Observation')
            plotter.plot_image(model.observation_variable.likelihood_dist.loc, 'Reconstruction')

        if done:
            # TODO: need a final collection of reward inside model
            # TODO: this final logging will not be correct because the model never sees this observation
            # logger.log_step(model, observation, reward)
            grads = grad_buffer.collect()
            episode_log = logger.log_episode(model, grads)
            plotter.plot(episode_log)
            print_episode_metrics(episode, episode_log)
            grad_buffer.update()

            episode += 1
            observation = env.reset()
            reward = None
            model.reset()

    return model
