import time
from .collect_episode import collect_episode
from .train_batch import train_batch
from .test_model import test_model
from .agent_kl import estimate_agent_kl


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
    critic_delay = args.critic_delay
    ckpt_interval = args.checkpoint_interval
    assert args.n_initial_steps >= buffer.batch_size * buffer.sequence_length

    while timestep < args.n_total_steps:
        # estimate agent KL (change in policy distribution)
        agent_kl = estimate_agent_kl(env, agent, buffer.last_episode)
        plotter.plot_agent_kl(agent_kl, timestep)
        # collect an episode
        print(' -- Collecting Episode: ' + str(n_episodes + 1))
        t_start = time.time()
        r = buffer.total_steps < n_initial_steps
        episode, episode_length, _ = collect_episode(env, agent, random=r)
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
            if 'state_likelihood_model' in dir(agent.q_value_estimator):
                # pre-train the model (and value function)
                if model_trained == False:
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

                # evaluate the model
                predictions, log_likelihoods = test_model(episode, agent)
                plotter.plot_model_eval(episode, predictions, log_likelihoods, timestep)

            print('Training...')
            # train the agent
            for update in range(n_updates):
                if timestep % eval_interval == 0:
                    # evaluation
                    print('Evaluating at Step: ' + str(timestep))
                    episode, _, eval_states = collect_episode(env, agent, eval=True)
                    plotter.log_eval(episode, eval_states, timestep)
                    print('Done.')
                if timestep % ckpt_interval == 0:
                    print('Checkpointing...')
                    plotter.save_checkpoint(timestep)
                    print('Done.')
                if update % 50 == 0:
                    print(' Batch: ' + str(update + 1) + ' of ' + str(n_updates) + '.')
                # update the critic (Q-network + model)
                for critic_update in range(critic_delay):
                    batch = buffer.sample()
                    results = train_batch(agent, batch, optimizer, critic_only=True)
                # actor (and critic) update
                t_start = time.time()
                batch = buffer.sample()
                results = train_batch(agent, batch, optimizer)
                t_end = time.time()
                if update % 50 == 0:
                    print('Duration: ' + '{:.2f}'.format(t_end - t_start) + ' s / batch.')
                plotter.log_results(results)
                timestep += 1
            plotter.plot_results(timestep)
