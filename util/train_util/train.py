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
    eval_interval = args.eval_interval

    while timestep < args.n_total_steps:
        # collect an episode
        print(' -- Collecting Episode: ' + str(n_episodes + 1))
        t_start = time.time()
        r = buffer.total_steps < n_initial_batches * buffer.batch_size * buffer.sequence_length
        episode, episode_length = collect_episode(env, agent, random=r)
        plotter.plot_episode(episode, timestep)
        t_end = time.time()
        print('Duration: ' + '{:.2f}'.format(t_end - t_start) + ' s.')
        # timestep += episode_length
        n_episodes += 1
        buffer.append(episode)
        n_updates = update_factor * episode_length

        # train on samples from buffer
        if buffer.total_steps >= n_initial_batches * buffer.batch_size * buffer.sequence_length:
            # pre-train the model for a pre-specified number of steps
            if 'horizon' in dir(agent.q_value_estimator) and model_trained == False:
                print('Pre-Training the model...')
                for update in range(n_pretrain_updates):
                    print(' Batch: ' + str(update + 1) + ' of ' + str(n_pretrain_updates) + '.')
                    t_start = time.time()
                    batch = buffer.sample()
                    results = train_batch(agent, batch, optimizer, model_only=True)
                    t_end = time.time()
                    print('Duration: ' + '{:.2f}'.format(t_end - t_start) + ' s.')
                model_trained = True
                plotter.save_checkpoint(timestep)

            print('Training...')
            # train the agent
            for update in range(n_updates):
                if timestep % eval_interval == 0:
                    # evaluation
                    print('Evaluating at Step: ' + str(timestep))
                    episode, _ = collect_episode(env, agent, eval=True)
                    plotter.plot_eval(episode, timestep)
                print(' Batch: ' + str(update + 1) + ' of ' + str(n_updates) + '.')
                t_start = time.time()
                batch = buffer.sample()
                results = train_batch(agent, batch, optimizer)
                t_end = time.time()
                print('Duration: ' + '{:.2f}'.format(t_end - t_start) + ' s.')
                plotter.log_results(results)
                timestep += 1
            plotter.plot_results(timestep)
            # if on_policy:
            #     buffer.empty()
