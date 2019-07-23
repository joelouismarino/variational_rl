import time
from .collect_episode import collect_episode
from .train_batch import train_batch


def train(agent, env, buffer, optimizer, plotter):
    """
    Collect episodes and train the agent.
    """
    timestep = 0
    n_episodes = 0
    n_initial_batches = 1
    n_updates = 1
    while timestep < 1e6:
        # collect an episode
        # print(logger.log_str + ' -- Collecting Episode: ' + str(n_episodes + 1))
        t_start = time.time()
        r = len(buffer) < n_initial_batches * buffer.batch_size
        episode, episode_length = collect_episode(env, agent, random=r)
        plotter.plot_episode(episode, timestep)
        t_end = time.time()
        print('Duration: ' + '{:.2f}'.format(t_end - t_start) + ' s.')
        timestep += episode_length
        n_episodes += 1
        buffer.append(episode)

        # train on samples from buffer
        if len(buffer) >= n_initial_batches * buffer.batch_size:
            print('Training Model.')
            for update in range(n_updates):
                print(' Batch: ' + str(update + 1))
                t_start = time.time()
                batch = buffer.sample()
                results = train_batch(agent, batch, optimizer)
                t_end = time.time()
                print('Duration: ' + '{:.2f}'.format(t_end - t_start) + ' s.')
                # results = flatten(results)
                print(timestep)
                plotter.log_results(results, timestep)
                # for n, m in results.items():
                #     experiment.log_metric(n, m, timestep)

            # if on_policy:
            #     buffer.empty()