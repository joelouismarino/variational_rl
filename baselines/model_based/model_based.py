import torch
from .config import get_model_args
from .models import Model
from .gradient_buffer import GradientBuffer
from .util import Viewer
from .util.plot_util import PlotVisdom
from .util.log_util import Logger
from .util.print_util import print_metrics


def learn(env, seed, total_timesteps, **kwargs):

    # torch.manual_seed(seed)

    plots = PlotVisdom()
    logger = Logger()

    model_args = get_model_args(env)
    model = Model(**model_args)
    model.training = True

    grad_buffer = GradientBuffer(model, lr=0.001, capacity=100, batch_size=5)

    # obs_viewer = Viewer()
    # recon_viewer = Viewer()

    observation = env.reset()
    reward = None
    n_episodes = 0

    for step_num in range(total_timesteps):

        # obs_viewer.view(observation)

        action = model.act(observation, reward)

        # recon_viewer.view(model.observation_variable.likelihood_dist.loc)

        logger.add_log(step_num, model, observation, reward)
        if step_num % 10 == 0:
            plots.plot_visdom(logger.log)
            plots.visualize_obs_visdom(observation)
            plots.visualize_recon_visdom(model.observation_variable.likelihood_dist.loc)

        print_metrics(step_num, n_episodes, model, observation, reward)

        observation, reward, done, _ = env.step(action)

        # env.render()

        grad_buffer.accumulate()

        if done:
            grad_buffer.collect()
            grad_buffer.update()
            n_episodes += 1
            observation = env.reset()
            reward = None
            model.reset()

    return model
