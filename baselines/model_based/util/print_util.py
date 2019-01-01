
def print_metrics(step_num, n_episodes, model, observation, reward):

    print('Step Num: ' + str(step_num))
    print('     Episode: ' + str(n_episodes+1))
    print('     Free Energy: ' + str(model.free_energy(observation, reward).item()))
    print('         KL Divergence: ' + str(model.kl_divergence().sum().item()))
    print('             State KL: ' + str(model.state_variable.kl_divergence().sum().item()))
    print('             Action KL: ' + str(model.action_variable.kl_divergence().sum().item()))
    print('         Cond. Log Likelihood: ' + str(model.cond_log_likelihood(observation, reward).item()))
    print('             Observation CLL: ' + str(model.observation_variable.cond_log_likelihood(observation).sum().item()))
    if reward is not None:
        print('             Reward CLL: ' + str(model.reward_variable.cond_log_likelihood(reward).sum().item()))
        print('             Optimality CLL: ' + str(reward))
