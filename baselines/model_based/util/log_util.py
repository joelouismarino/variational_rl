
log_items = ['Step', 'Free Energy', 'KL', 'State KL', 'Action KL', 'CLL',
                 'Obs CLL', 'Reward CLL', 'Optimality CLL', 'State Inf. Improvement']


class Logger:

    def __init__(self):
        self.log = {}
        self._init_log()

    def _init_log(self):
        for var in log_items:
            self.log[var] = []

    def add_log(self, step_num, model, observation, reward):
        self.log['Step'].append(step_num)
        self.log['Free Energy'].append(model.free_energy(observation, reward).item())
        self.log['KL'].append(model.kl_divergence().sum().item())
        self.log['State KL'].append(model.state_variable.kl_divergence().sum().item())
        self.log['Action KL'].append(model.action_variable.kl_divergence().sum().item())
        self.log['CLL'].append(model.cond_log_likelihood(observation, reward).item())
        self.log['Obs CLL'].append(model.observation_variable.cond_log_likelihood(observation).sum().item())
        if reward is not None:
            self.log['Reward CLL'].append(model.reward_variable.cond_log_likelihood(reward).sum().item())
            self.log['Optimality CLL'].append(reward)
        else:
            self.log['Reward CLL'].append(0)
            self.log['Optimality CLL'].append(0)
        state_inf_improvement = model.state_inf_free_energies[0] - model.state_inf_free_energies[-1]
        state_inf_improvement /= model.state_inf_free_energies[0]
        state_inf_improvement *= 100.
        self.log['State Inf. Improvement'].append(state_inf_improvement.item())
