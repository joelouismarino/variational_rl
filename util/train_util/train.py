import time
from .collect_episode import collect_episode
from .train_batch import train_batch
from .test_model import test_model


def train(agent, env, buffer, optimizer, plotter, args):
    """
    Collect episodes and train the agent.
    """
    timestep = 0
    n_episodes = 0
    n_initial_steps = args.n_initial_steps
    update_factor = args.update_factor
    model_trained = False
    n_pretrain_updates = args.n_pretrain_updates
    eval_interval = args.eval_interval
    assert args.n_initial_steps >= buffer.batch_size * buffer.sequence_length

    while timestep < args.n_total_steps:
        # collect an episode
        print(' -- Collecting Episode: ' + str(n_episodes + 1))
        t_start = time.time()
        r = buffer.total_steps < n_initial_steps
        episode, episode_length = collect_episode(env, agent, random=r)
        plotter.plot_episode(episode, timestep)
        t_end = time.time()
        print('Duration: ' + '{:.2f}'.format(t_end - t_start) + ' s.')
        # timestep += episode_length
        n_episodes += 1
        buffer.append(episode)
        n_updates = update_factor * episode_length

        # train on samples from buffer
        if buffer.total_steps >= n_initial_steps:
            # model pre-training and evaluation
            if 'horizon' in dir(agent.q_value_estimator):
                # pre-train the model (and value function)
                # if True:
                if model_trained == False:
                    print('Pre-Training the model...')
                    # state_clls = []
                    # reward_clls = []
                    # q_losses = []
                    # update = 0
                    # min_state_cll = -100000
                    # patience_steps = 0
                    # still_training = True
                    # while still_training:
                    for update in range(n_pretrain_updates):
                        # print(' Batch: ' + str(update + 1) + '.')
                        print(' Batch: ' + str(update + 1) + ' of ' + str(n_pretrain_updates) + '.')
                        t_start = time.time()
                        batch = buffer.sample()
                        results = train_batch(agent, batch, optimizer, model_only=True)
                        # state_clls.append(results['state_cll'])
                        # reward_clls.append(results['reward_cll'])
                        # q_losses.append(results['q_loss1'])
                        t_end = time.time()
                        print('Duration: ' + '{:.2f}'.format(t_end - t_start) + ' s.')
                        # print('State CLL: ' + str(results['state_cll']))
                        # print('Patience: ' + str(patience_steps))
                        # if results['state_cll'] > min_state_cll:
                        #     min_state_cll = results['state_cll']
                        #     patience_steps = 0
                        # else:
                        #     patience_steps += 1
                        #     if patience_steps > 50:
                        #         still_training = False
                        # update += 1
                    # import ipdb; ipdb.set_trace()
                    model_trained = True
                    plotter.save_checkpoint(timestep)

                # evaluate the model
                predictions, log_likelihoods = test_model(episode, agent)
                plotter.plot_model_eval(episode, predictions, log_likelihoods, timestep)

            print('Training...')
            # train the agent
            for update in range(n_updates):
                if timestep % eval_interval == 0:
                    # evaluation
                    print('Evaluating at Step: ' + str(timestep))
                    episode, _ = collect_episode(env, agent, eval=True)
                    plotter.log_eval(episode, timestep)
                    print('Done.')
                print(' Batch: ' + str(update + 1) + ' of ' + str(n_updates) + '.')
                t_start = time.time()
                batch = buffer.sample()
                results = train_batch(agent, batch, optimizer)
                t_end = time.time()
                print('Duration: ' + '{:.2f}'.format(t_end - t_start) + ' s.')
                plotter.log_results(results)
                timestep += 1
            plotter.plot_results(timestep)
