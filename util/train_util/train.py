import time
from .collect_episode import collect_episode
from .train_batch import train_batch


def train(agent, env, buffer, optimizer, plotter, args):
    """
    Collect episodes and train the agent.
    """
    timestep = 0
    n_episodes = 0
    n_initial_batches = args.n_initial_batches
    update_factor = args.update_factor
    model_trained = False
    n_pretrain_updates = args.n_pretrain_updates

    while timestep < args.n_total_steps:
        # collect an episode
        print(' -- Collecting Episode: ' + str(n_episodes + 1))
        t_start = time.time()
        r = buffer.total_steps < n_initial_batches * buffer.batch_size * buffer.sequence_length
        episode, episode_length = collect_episode(env, agent, random=r)
        plotter.plot_episode(episode, timestep)
        t_end = time.time()
        print('Duration: ' + '{:.2f}'.format(t_end - t_start) + ' s.')
        timestep += episode_length
        n_episodes += 1
        buffer.append(episode)
        n_updates = update_factor * episode_length

        # train on samples from buffer
        if buffer.total_steps >= n_initial_batches * buffer.batch_size * buffer.sequence_length:
            # pre-train the model for a pre-specified number of steps
            if model_trained == False and agent.type == 'model_based' and agent.rollout_length > 0:
                print('Pre-Training the model...')
                agent.train_model_only = True
                for update in range(n_pretrain_updates):
                    print(' Batch: ' + str(update + 1) + ' of ' + str(n_pretrain_updates) + '.')
                    t_start = time.time()
                    batch = buffer.sample()
                    results = train_batch(agent, batch, optimizer, model_only=True)
                    # if update % 1000 == 0:
                    #     plotter.log_results(results, update)
                    t_end = time.time()
                    print('Duration: ' + '{:.2f}'.format(t_end - t_start) + ' s.')
                agent.train_model_only = False
                model_trained = True
                plotter.save_checkpoint(timestep)

            print('Training...')
            # train the agent
            # for update in range(n_updates):
            for update in range(5):
                print(' Batch: ' + str(update + 1) + ' of ' + str(n_updates) + '.')
                t_start = time.time()
                batch = buffer.sample()
                results = train_batch(agent, batch, optimizer)
                t_end = time.time()
                print('Duration: ' + '{:.2f}'.format(t_end - t_start) + ' s.')
                plotter.log_results(results)
            plotter.plot_results(timestep)
            # if on_policy:
            #     buffer.empty()
