import torch
from .config import get_agent_args
from .agents import get_agent
from .buffers import DataBuffer
from .optimizers import Optimizer
from .util.plot_util import Plotter
from .util.log_util import Logger
from .util.print_util import print_step_metrics, print_episode_metrics
from .util.train_util import collect_episode, train


def learn(env, seed, total_timesteps, log_dir, batch_size=16, n_updates=50,
          n_initial_batches=1, train_seq_len=32, lr=1e-4, device=None,
          ckpt_path=None, **kwargs):

    # torch.manual_seed(seed)

    # create the agent
    agent_args = get_agent_args(env)
    agent = get_agent(agent_args)
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path)
        agent.load(state_dict)
    if device is not None:
        agent.to(device)
    agent.reset()

    # create the buffer
    buffer = DataBuffer(batch_size=batch_size, sequence_length=train_seq_len)

    # create the optimizer
    base_lr = lr
    lr = {'state_inference_model': base_lr,
          'action_inference_model': base_lr,
          'state_prior_model': base_lr,
          'action_prior_model': base_lr,
          'obs_likelihood_model': base_lr,
          'reward_likelihood_model': base_lr,
          'done_likelihood_model': base_lr,
          'value_model': base_lr}
    optimizer = Optimizer(agent, lr=lr, norm_grad=0.5)

    # logging and plotting
    if hasattr(env, 'spec'):
        env_name = env.spec.id
    else:
        env_name = env.venv.envs[0].spec.id
    exp_args = {'env': env_name, 'seed': seed,
                'total_timesteps': total_timesteps, 'batch_size': batch_size,
                'n_updates': n_updates, 'n_initial_batches': n_initial_batches,
                'lr': lr, 'device': device, 'agent_args': agent_args}
    logger = Logger(log_dir, exp_args, agent)
    exp_args['log_str'] = logger.log_str
    plotter = Plotter(log_dir, exp_args)

    # collect episodes and train
    timestep = 0
    n_episodes = 0
    while timestep < total_timesteps:

        # collect an episode
        print(logger.log_str + ' -- Collecting Episode: ' + str(n_episodes + 1))
        episode, episode_length = collect_episode(env, agent)
        timestep += episode_length
        n_episodes += 1
        plotter.plot_episode(episode)
        logger.log_episode(episode)
        buffer.append(episode)

        # train on samples from buffer
        if len(buffer) >= n_initial_batches * batch_size:
            print('Training Model.')
            for update in range(n_updates):
                print(' Batch: ' + str(update + 1))
                batch = buffer.sample()
                results = train(agent, batch, optimizer)
                logger.log_train_step(results)
                plotter.plot_train_step(results, plot=(update==n_updates-1))

    return agent
