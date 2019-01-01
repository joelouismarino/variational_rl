
def init_log(log_items):
    log = {}
    for var in log_items:
        log[var] = []
    return log

def add_log(log, step_num, model, observation, reward):
    log['Step'].append(step_num)
    log['Free Energy'].append(model.free_energy(observation, reward).item())
    log['KL'].append(model.kl_divergence().sum().item())
    log['State KL'].append(model.state_variable.kl_divergence().sum().item())
    log['Action KL'].append(model.action_variable.kl_divergence().sum().item())
    log['CLL'].append(model.cond_log_likelihood(observation, reward).item())
    log['Obs CLL'].append(model.observation_variable.cond_log_likelihood(observation).sum().item())
    if reward is not None:
        log['Reward CLL'].append(model.reward_variable.cond_log_likelihood(reward).sum().item())
        log['Optimality CLL'].append(reward)
    else:
        log['Reward CLL'].append(0)
        log['Optimality CLL'].append(0)
    return log
