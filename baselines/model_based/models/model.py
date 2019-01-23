import torch
import torch.nn as nn
import torch.distributions as dist
from ..modules.networks import get_network
from ..modules.variables import get_variable
from ..misc import clear_gradients, one_hot_to_index


class Model(nn.Module):
    """
    Generative Model-Based Agent
    """
    def __init__(self, state_variable_args, action_variable_args,
                 observation_variable_args, reward_variable_args,
                 done_variable_args, state_prior_args, action_prior_args,
                 obs_likelihood_args, reward_likelihood_args,
                 done_likelihood_args, state_inference_args,
                 action_inference_args, misc_args):
        super(Model, self).__init__()

        # networks
        self.state_prior_model = get_network(state_prior_args)
        self.action_prior_model = get_network(action_prior_args)
        self.obs_likelihood_model = get_network(obs_likelihood_args)
        self.reward_likelihood_model = get_network(reward_likelihood_args)
        self.done_likelihood_model = get_network(done_likelihood_args)
        self.state_inference_model = get_network(state_inference_args)
        self.action_inference_model = get_network(action_inference_args)

        # variables
        state_variable_args['n_input'] = (self.state_prior_model.n_out,
                                          self.state_inference_model.n_out)
        self.state_variable = get_variable(latent=True, args=state_variable_args)

        action_variable_args['n_input'] = (self.action_prior_model.n_out,
                                           self.action_inference_model.n_out)
        self.action_variable = get_variable(latent=True, args=action_variable_args)

        observation_variable_args['n_input'] = self.obs_likelihood_model.n_out
        self.observation_variable = get_variable(latent=False, args=observation_variable_args)

        reward_variable_args['n_input'] = self.reward_likelihood_model.n_out
        self.reward_variable = get_variable(latent=False, args=reward_variable_args)

        done_variable_args['n_input'] = self.done_likelihood_model.n_out
        self.done_variable = get_variable(latent=False, args=done_variable_args)

        # miscellaneous
        self.n_inf_iter = misc_args['n_inf_iter']
        self.optimality_scale = misc_args['optimality_scale']

        # mode (either 'train' or 'eval')
        self._mode = 'train'

        # stores the variables during an episode
        self.episode = {'observation': [], 'reward': [], 'done': [],
                        'state': [], 'action': [], 'reconstruction': [],
                        'prediction': []}
        # stores the objectives during an episode
        self.objectives = {'observation': [], 'reward': [], 'optimality': [],
                           'done': [], 'state': [], 'action': []}
        # stores inference improvement
        self.inference_improvement = {'state': [], 'action': []}
        # stores the log probabilities during an episode
        self.log_probs = {'action': []}

        self.valid = []

        self._prev_action = None

        self.obs_reconstruction = None
        self.obs_prediction = None
        self.batch_size = 1

    def act(self, observation, reward=None, done=False, action=None, valid=None):
        observation, reward, action, done, valid = self._change_device(observation, reward, action, done, valid)
        self.step_state()
        self.state_inference(observation, reward, done, valid)
        self.generate_observation()
        self.generate_reward()
        self.generate_done()
        self.step_action()
        self.action_inference()
        if self._mode == 'train':
            self._collect_objectives_and_log_probs(observation, reward, done, action, valid)
            self._prev_action = action
            self.valid.append(valid)
        else:
            self._collect_episode(observation, reward, done)
            if observation is not None:
                action = self.action_variable.sample()
                action = self._convert_action(action).cpu().numpy()
        return action

    # def state_inference(self, observation, reward, done, valid):
    #     self.inference_mode()
    #     # infer the approx. posterior on the state
    #     self.state_variable.init_approx_post()
    #     self.generate_observation()
    #     self.generate_reward()
    #     self.generate_done()
    #     self.obs_prediction = self.observation_variable.likelihood_dist.loc.detach()
    #     obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum(dim=(2,3)).sum(dim=1, keepdim=True)
    #     reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward).sum(dim=1, keepdim=True)
    #     done_log_likelihood = self.done_variable.cond_log_likelihood(done).sum(dim=1, keepdim=True)
    #     state_kl = self.state_variable.kl_divergence().sum(dim=1, keepdim=True)
    #     state_inf_free_energy = state_kl - (1 - done) * obs_log_likelihood - reward_log_likelihood - done_log_likelihood
    #     state_inf_free_energy = valid * state_inf_free_energy
    #     initial_free_energy = state_inf_free_energy
    #
    #     # infer the approx. posterior on the state
    #     inf_input = observation - 0.5
    #     inf_input = self.state_inference_model(inf_input)
    #     self.state_variable.infer(inf_input)
    #     # final evaluation
    #     self.generate_observation()
    #     self.generate_reward()
    #     self.generate_done()
    #     obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum(dim=(2,3)).sum(dim=1, keepdim=True)
    #     reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward).sum(dim=1, keepdim=True)
    #     done_log_likelihood = self.done_variable.cond_log_likelihood(done).sum(dim=1, keepdim=True)
    #     state_kl = self.state_variable.kl_divergence().sum(dim=1, keepdim=True)
    #     state_inf_free_energy = state_kl - (1 - done) * obs_log_likelihood - reward_log_likelihood - done_log_likelihood
    #     state_inf_free_energy = valid * state_inf_free_energy
    #     final_free_energy = state_inf_free_energy
    #     inference_improvement = torch.zeros(initial_free_energy.shape).to(self.device)
    #     valid_inds = torch.nonzero(valid[:,0])
    #     inference_improvement[valid_inds] = initial_free_energy[valid_inds] - final_free_energy[valid_inds]
    #     inference_improvement[valid_inds] = 100 * inference_improvement[valid_inds] / initial_free_energy[valid_inds]
    #     self.inference_improvement['state'].append(inference_improvement)
    #     (state_inf_free_energy.sum()).backward(retain_graph=True)
    #     clear_gradients(self.generative_parameters())
    #     self.generative_mode()
    #     self.obs_reconstruction = self.observation_variable.likelihood_dist.loc.detach()

    def state_inference(self, observation, reward, done, valid):
        self.inference_mode()
        # infer the approx. posterior on the state
        self.state_variable.init_approx_post()
        for inf_iter in range(self.n_inf_iter):
            # evaluate conditional log likelihood of observation and state KL divergence
            self.generate_observation()
            self.generate_reward()
            self.generate_done()
            obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum(dim=(2,3)).sum(dim=1, keepdim=True)
            reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward).sum(dim=1, keepdim=True)
            done_log_likelihood = self.done_variable.cond_log_likelihood(done).sum(dim=1, keepdim=True)
            state_kl = self.state_variable.kl_divergence().sum(dim=1, keepdim=True)
            state_inf_free_energy = state_kl - (1 - done) * obs_log_likelihood - reward_log_likelihood - done_log_likelihood
            state_inf_free_energy = valid * state_inf_free_energy
            if inf_iter == 0:
                self.obs_prediction = self.observation_variable.likelihood_dist.loc.detach()
                initial_free_energy = state_inf_free_energy

            (state_inf_free_energy.sum()).backward(retain_graph=True)
            # update approx. posterior
            inf_input = self.state_variable.params_and_grads()
            inf_input = self.state_inference_model(inf_input)
            self.state_variable.infer(inf_input)
        # final evaluation
        self.generate_observation()
        self.generate_reward()
        self.generate_done()
        obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum(dim=(2,3)).sum(dim=1, keepdim=True)
        reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward).sum(dim=1, keepdim=True)
        done_log_likelihood = self.done_variable.cond_log_likelihood(done).sum(dim=1, keepdim=True)
        state_kl = self.state_variable.kl_divergence().sum(dim=1, keepdim=True)
        state_inf_free_energy = state_kl - (1 - done) * obs_log_likelihood - reward_log_likelihood - done_log_likelihood
        state_inf_free_energy = valid * state_inf_free_energy
        final_free_energy = state_inf_free_energy
        inference_improvement = torch.zeros(initial_free_energy.shape).to(self.device)
        valid_inds = torch.nonzero(valid[:,0])
        inference_improvement[valid_inds] = initial_free_energy[valid_inds] - final_free_energy[valid_inds]
        inference_improvement[valid_inds] = 100 * inference_improvement[valid_inds] / initial_free_energy[valid_inds]
        self.inference_improvement['state'].append(inference_improvement)
        (state_inf_free_energy.sum()).backward(retain_graph=True)
        clear_gradients(self.generative_parameters())
        self.generative_mode()
        self.obs_reconstruction = self.observation_variable.likelihood_dist.loc.detach()

    def action_inference(self, action=None):
        self.inference_mode()
        # infer the approx. posterior on the action
        # self.action_variable.init_approx_post()
        # TODO: implement planning inference
        hidden_state = self.state_prior_model.state
        state = self.state_variable.sample()
        if self._prev_action is not None:
            action = self._prev_action
        else:
            action = self.action_variable.sample()
        inf_input = self.action_inference_model(torch.cat((state, hidden_state, action), dim=1))
        self.action_variable.infer(inf_input)
        # clear_gradients(self.generative_parameters())
        self.generative_mode()

    def step_state(self):
        # calculate the prior on the state variable
        if not self.state_variable.reinitialized:
            state = self.state_variable.sample()
            if self._prev_action is not None:
                action = self._prev_action
            else:
                action = self.action_variable.sample()
            prior_input = self.state_prior_model(torch.cat((state, action), dim=1))
            self.state_variable.step(prior_input)

    def step_action(self):
        # calculate the prior on the action variable
        if not self.action_variable.reinitialized:
            hidden_state = self.state_prior_model.state
            state = self.state_variable.sample()
            if self._prev_action is not None:
                action = self._prev_action
            else:
                action = self.action_variable.sample()
            prior_input = self.action_prior_model(torch.cat((state, hidden_state, action), dim=1))
            self.action_variable.step(prior_input)

    def generate_observation(self):
        # generate the conditional likelihood for the observation
        hidden_state = self.state_prior_model.state
        state = self.state_variable.sample()
        # likelihood_input = self.obs_likelihood_model(state)
        likelihood_input = self.obs_likelihood_model(torch.cat([hidden_state, state], dim=1))
        self.observation_variable.generate(likelihood_input)

    def generate_reward(self):
        # generate the conditional likelihood for the reward
        hidden_state = self.state_prior_model.state
        state = self.state_variable.sample()
        likelihood_input = self.reward_likelihood_model(torch.cat([hidden_state, state], dim=1))
        # likelihood_input = self.reward_likelihood_model(state)
        self.reward_variable.generate(likelihood_input)

    def generate_done(self):
        # generate the conditional likelihood for episode being done
        hidden_state = self.state_prior_model.state
        state = self.state_variable.sample()
        likelihood_input = self.done_likelihood_model(torch.cat([hidden_state, state], dim=1))
        self.done_variable.generate(likelihood_input)

    def free_energy(self, observation, reward, done):
        observation, reward, done = self._change_device(observation, reward, done)
        cond_log_likelihood = self.cond_log_likelihood(observation, reward, done)
        kl_divergence = self.kl_divergence()
        free_energy = kl_divergence - cond_log_likelihood
        return free_energy

    def cond_log_likelihood(self, observation, reward, done):
        observation, reward, done = self._change_device(observation, reward, done)
        obs_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum(dim=(2,3)).sum(dim=1, keepdim=True)
        done_log_likelihood = self.done_variable.cond_log_likelihood(done).sum(dim=1, keepdim=True)
        reward_log_likelihood = opt_log_likelihood = 0.
        if reward is not None:
            reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward).sum(dim=1, keepdim=True)
            opt_log_likelihood = self.optimality_scale * (reward - 1.)
        cond_log_likelihood = obs_log_likelihood + reward_log_likelihood + opt_log_likelihood + done_log_likelihood
        return cond_log_likelihood

    def kl_divergence(self):
        state_kl_divergence = self.state_variable.kl_divergence().sum(dim=1, keepdim=True)
        action_kl_divergence = self.action_variable.kl_divergence().view(-1, 1)
        kl_divergence = state_kl_divergence + action_kl_divergence
        return kl_divergence

    def evaluate(self):
        # evaluate the objective, averaged over the batch, backprop

        results = {}
        n_valid_steps = torch.stack(self.valid).sum(dim=0).sub(1)
        # average objectives over time and batch for reporting
        for objective_name, objective in self.objectives.items():
            obj = torch.stack(objective).sum(dim=0).div(n_valid_steps).mean(dim=0)
            if objective_name in ['observation', 'reward', 'optimality', 'done']:
                obj = obj * -1
            results[objective_name] = obj.detach().cpu().item()

        for name, improvement in self.inference_improvement.items():
            if len(improvement) > 0:
                imp = torch.stack(improvement).sum(dim=0).div(n_valid_steps).mean(dim=0)
                results[name + '_improvement'] = imp.detach().cpu().item()

        # sum the objectives
        n_steps = len(self.objectives['observation'])
        free_energy = torch.zeros(n_steps, self.batch_size, 1).to(self.device)
        for objective_name, objective in self.objectives.items():
            free_energy = free_energy + torch.stack(objective)

        # calculate the reinforce terms
        future_sums = torch.flip(torch.cumsum(torch.flip(free_energy.detach(), dims=[0]), dim=0), dims=[0])
        # if future_sums.shape[0] > 2:
        #     future_sums = (future_sums - future_sums[1:].mean()) / (future_sums[1:].std() + 1e-6)
        log_probs = torch.stack(self.log_probs['action'])
        reinforce_terms = - log_probs * future_sums
        # free_energy = free_energy + reinforce_terms

        # time average
        free_energy = free_energy.sum(dim=0).div(n_valid_steps)

        # batch average
        free_energy = free_energy.mean(dim=0)

        # backprop
        free_energy.sum().backward()

        return results


    def _collect_objectives_and_log_probs(self, observation, reward, done, action, valid):
        done_log_likelihood = self.done_variable.cond_log_likelihood(done).sum(dim=1, keepdim=True)
        self.objectives['done'].append(-done_log_likelihood * valid)

        reward_log_likelihood = self.reward_variable.cond_log_likelihood(reward).sum(dim=1, keepdim=True)
        self.objectives['reward'].append(-reward_log_likelihood * valid)

        optimality_log_likelihood = self.optimality_scale * (reward - 1.)
        self.objectives['optimality'].append(-optimality_log_likelihood * valid)

        observation_log_likelihood = self.observation_variable.cond_log_likelihood(observation).sum(dim=(2,3)).sum(dim=1, keepdim=True)
        self.objectives['observation'].append(-observation_log_likelihood * (1 - done) * valid)

        state_kl = self.state_variable.kl_divergence().sum(dim=1, keepdim=True)
        self.objectives['state'].append(state_kl * (1 - done) * valid)

        action_kl = self.action_variable.kl_divergence().view(-1, 1)
        self.objectives['action'].append(action_kl * valid)

        action_ind = self._convert_action(action)
        action_log_prob = self.action_variable.approx_post_dist.log_prob(action_ind).view(-1, 1)
        self.log_probs['action'].append(action_log_prob * valid)

    def _collect_episode(self, observation, reward, done):
        if not done:
            self.episode['observation'].append(observation)
            self.episode['reconstruction'].append(self.obs_reconstruction)
            self.episode['prediction'].append(self.obs_prediction)
            self.episode['action'].append(self.action_variable.sample())
            self.episode['state'].append(self.state_variable.sample())
        else:
            obs = self.episode['observation'][0]
            action = self.episode['action'][0]
            state = self.episode['state'][0]
            self.episode['observation'].append(obs.new(obs.shape).zero_())
            self.episode['action'].append(action.new(action.shape).zero_())
            self.episode['state'].append(state.new(state.shape).zero_())
        self.episode['reward'].append(reward)
        self.episode['done'].append(done)

    def _convert_action(self, action):
        # converts categorical action from one-hot encoding to the action index
        if self.action_variable.approx_post_dist_type == getattr(torch.distributions, 'Categorical'):
            action = one_hot_to_index(action)
        return action

    def _change_device(self, observation, reward, action, done, valid):
        if observation is None:
            observation = torch.zeros(self.episode['observation'][0].shape)
        if observation.device != self.device:
            observation = observation.to(self.device)
        if type(reward) == float:
            reward = torch.tensor(reward).to(torch.float32).view(1, 1)
        if reward.device != self.device:
            reward = reward.to(self.device)
        if action is not None:
            if action.device != self.device:
                action = action.to(self.device)
        if type(done) == bool:
            done = torch.tensor(done).to(torch.float32).view(1, 1)
        if done.device != self.device:
            done = done.to(self.device)
        if valid is None:
            valid = torch.ones(done.shape[0], 1)
        if valid.device != self.device:
            valid = valid.to(self.device)
        return observation, reward, action, done, valid

    def get_episode(self):
        """
        Concatenate each variable in the episode. Put on the CPU.
        """
        episode = {}
        for k, v in self.episode.items():
            episode[k] = torch.cat(v, dim=0).cpu()
        return episode

    def reset(self, batch_size=1):
        # reset the variables
        self.state_variable.reset(batch_size)
        self.action_variable.reset(batch_size)
        self.observation_variable.reset()
        self.reward_variable.reset()
        self.done_variable.reset()

        # reset the networks
        self.state_prior_model.reset(batch_size)
        self.action_prior_model.reset(batch_size)
        self.obs_likelihood_model.reset()
        self.reward_likelihood_model.reset()
        self.done_likelihood_model.reset()

        # reset the episode, objectives, and log probs
        self.episode = {'observation': [], 'reward': [], 'done': [],
                        'state': [], 'action': [], 'reconstruction': [],
                        'prediction': []}
        self.objectives = {'observation': [], 'reward': [], 'optimality': [],
                           'done': [], 'state': [], 'action': []}
        self.inference_improvement = {'state': [], 'action': []}
        self.log_probs = {'action': []}

        self._prev_action = None
        self.valid = []
        self.batch_size = batch_size
        self.state_inf_free_energies = []
        self.obs_reconstruction = None
        self.obs_prediction = None

    @property
    def device(self):
        return self.inference_parameters()[0].device

    def train(self, *args):
        super(Model, self).train(*args)
        self._mode = 'train'

    def eval(self, *args):
        super(Model, self).eval(*args)
        self._mode = 'eval'

    def parameters(self):
        param_dict = {}

        param_dict['state_inference_model'] = nn.ParameterList()
        param_dict['state_inference_model'].extend(list(self.state_inference_model.parameters()))
        param_dict['state_inference_model'].extend(list(self.state_variable.inference_parameters()))

        param_dict['action_inference_model'] = nn.ParameterList()
        param_dict['action_inference_model'].extend(list(self.action_inference_model.parameters()))
        param_dict['action_inference_model'].extend(list(self.action_variable.inference_parameters()))

        param_dict['state_prior_model'] = nn.ParameterList()
        param_dict['state_prior_model'].extend(list(self.state_prior_model.parameters()))
        param_dict['state_prior_model'].extend(list(self.state_variable.generative_parameters()))

        param_dict['action_prior_model'] = nn.ParameterList()
        param_dict['action_prior_model'].extend(list(self.action_prior_model.parameters()))
        param_dict['action_prior_model'].extend(list(self.action_variable.generative_parameters()))

        param_dict['obs_likelihood_model'] = nn.ParameterList()
        param_dict['obs_likelihood_model'].extend(list(self.obs_likelihood_model.parameters()))
        param_dict['obs_likelihood_model'].extend(list(self.observation_variable.parameters()))

        param_dict['reward_likelihood_model'] = nn.ParameterList()
        param_dict['reward_likelihood_model'].extend(list(self.reward_likelihood_model.parameters()))
        param_dict['reward_likelihood_model'].extend(list(self.reward_variable.parameters()))

        param_dict['done_likelihood_model'] = nn.ParameterList()
        param_dict['done_likelihood_model'].extend(list(self.done_likelihood_model.parameters()))
        param_dict['done_likelihood_model'].extend(list(self.done_variable.parameters()))

        return param_dict

    def inference_parameters(self):
        params = nn.ParameterList()
        params.extend(list(self.state_inference_model.parameters()))
        params.extend(list(self.action_inference_model.parameters()))
        params.extend(list(self.state_variable.inference_parameters()))
        params.extend(list(self.action_variable.inference_parameters()))
        return params

    def generative_parameters(self):
        params = nn.ParameterList()
        params.extend(list(self.state_prior_model.parameters()))
        params.extend(list(self.action_prior_model.parameters()))
        params.extend(list(self.obs_likelihood_model.parameters()))
        params.extend(list(self.reward_likelihood_model.parameters()))
        params.extend(list(self.done_likelihood_model.parameters()))
        params.extend(list(self.state_variable.generative_parameters()))
        params.extend(list(self.action_variable.generative_parameters()))
        params.extend(list(self.observation_variable.parameters()))
        params.extend(list(self.reward_variable.parameters()))
        params.extend(list(self.done_variable.parameters()))
        return params

    def inference_mode(self):
        self.state_variable.inference_mode()
        self.action_variable.inference_mode()
        self.obs_likelihood_model.detach_hidden_state()
        self.reward_likelihood_model.detach_hidden_state()
        self.done_likelihood_model.detach_hidden_state()

    def generative_mode(self):
        self.state_variable.generative_mode()
        self.action_variable.generative_mode()
        self.obs_likelihood_model.attach_hidden_state()
        self.reward_likelihood_model.attach_hidden_state()
        self.done_likelihood_model.attach_hidden_state()
