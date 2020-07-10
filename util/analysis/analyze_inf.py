import comet_ml
import json
import torch
import copy
import numpy as np
from torch import optim
from lib import create_agent
from lib.distributions import kl_divergence
from util.env_util import create_env
from util.plot_util import load_checkpoint
from util.train_util import collect_episode
from misc import clear_gradients
from lib.value_estimators import get_value_estimator
from local_vars import PROJECT_NAME, WORKSPACE, LOADING_API_KEY, LOGGING_API_KEY

# load a checkpoint
# collect one or more episodes
# visualize inference over multiple seeds
# analyze inference performance (improvement, gap?)

CKPT_SUBSAMPLE = 5

def analyze_inference(exp_key, n_states, n_inf_seeds, n_action_samples=None,
                      ckpt_timestep=None, device_id=None):
    """
    Analyzes the inference procedure of a cached experiment. Performs inference
    on a number of states and returns the parameter estimates, value estimates,
    etc.

    Args:
        exp_key (str): the string of the comet experiment key
        n_states (int): number of states to evaluate
        n_inf_seeds (int): number of times to perform inference
        n_action_samples(int, optional): number of action samples to use for inference
                                         (None for agent default)
        ckpt_timestep (int, optional): checkpoint to load (None for final checkpoint)
        device_id (int): device to run on

    Returns dictionary containing:
        states: [n_states, n_state_dims]
        actions: [n_states, n_inf_seeds, n_inf_iters, n_action_samples, n_action_dims]
        value_estimates: [n_states, n_inf_seeds, n_inf_iters, n_action_samples, 1]
        params: loc: [n_states, n_inf_seeds, n_inf_iters, n_action_dims],
                scales: [n_states, n_inf_seeds, n_inf_iters, n_action_dims]
        grads: loc: [n_states, n_inf_seeds, n_inf_iters, n_action_dims],
               scales: [n_states, n_inf_seeds, n_inf_iters, n_action_dims]
    """
    # load the experiment
    comet_api = comet_ml.API(api_key=LOADING_API_KEY)
    experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                          workspace=WORKSPACE,
                                          experiment=exp_key)

    # create the environment
    param_summary = experiment.get_parameters_summary()
    env_name = [a for a in param_summary if a['name'] == 'env'][0]['valueCurrent']
    env = create_env(env_name)

    # create the agent
    asset_list = experiment.get_asset_list()
    agent_config_asset_list = [a for a in asset_list if 'agent_args' in a['fileName']]
    agent_args = None
    if len(agent_config_asset_list) > 0:
        # if we've saved the agent config dict, load it
        agent_args = experiment.get_asset(agent_config_asset_list[0]['assetId'])
        agent_args = json.loads(agent_args)
        agent_args = agent_args if 'opt_type' in agent_args['inference_optimizer_args'] else None
    agent = create_agent(env, agent_args=agent_args, device_id=device_id)[0]

    # load the checkpoint
    load_checkpoint(agent, exp_key, ckpt_timestep)

    # collect samples, perform inference
    if agent_args['approx_post_args']['update'] == 'direct' and n_inf_seeds != 1:
        print('Direct inference model detected. Using 1 inference seed.')
        n_inf_seeds = 1

    inf_iter_list = [a for a in param_summary if a['name'] == 'inference_optimizer_args_n_inf_iters']
    n_inf_iters = 1
    if len(inf_iter_list) > 0:
        n_inf_iters = int(inf_iter_list[0]['valueCurrent'])
    n_state_dims = env.observation_space.shape[-1]
    n_action_dims = env.action_space.shape[0]
    if n_action_samples is None:
        n_action_samples = agent.n_action_samples

    states = np.zeros((n_states, n_state_dims))
    # actions = np.array((n_states, n_inf_seeds, n_action_samples, n_action_dims))
    value_estimates = np.zeros((n_states, n_inf_seeds, n_action_samples, 1))
    if agent_args['approx_post_args']['update'] == 'iterative':
        it_value_estimates = np.zeros((n_states, n_inf_seeds, n_inf_iters, 1))
    if agent.state_value_estimator is not None:
        value_net_estimates = np.zeros((n_states, 1))
    params = {'loc': np.zeros((n_states, n_inf_seeds, n_action_dims)),
              'scale': np.zeros((n_states, n_inf_seeds, n_action_dims))}
    # grads = {'loc': np.array((n_states, n_inf_seeds, n_action_dims)),
    #          'scale': np.array((n_states, n_inf_seeds, n_action_dims))}
    print('Collecting ' + str(n_states) + ' states...')
    state = env.reset(); reward = 0; done = False
    for state_ind in range(n_states):
        if state_ind % 10 == 0:
            print(' State ' + str(state_ind) + '.')
        states[state_ind] = state[0]

        for inf_seed in range(n_inf_seeds):
            agent.reset(batch_size=1); agent.eval()
            action = agent.act(state, reward, done)
            # save the quantities
            # actions[state_ind, inf_seed, ] = action
            # get the state-value estimate from Q-net - KL
            if agent_args['approx_post_args']['update'] == 'iterative':
                value_ests = [-obj.detach().cpu().numpy() for obj in agent.inference_optimizer.estimated_objectives]
                it_value_estimates[state_ind, inf_seed] = np.stack(value_ests).reshape(-1, 1)
            on_policy_action = agent.approx_post.sample(n_action_samples)
            value_est = agent.estimate_objective(state, on_policy_action).view(n_action_samples, 1)
            value_estimates[state_ind, inf_seed] = value_est.detach().cpu().numpy()
            # get the state value estimate from the network
            if agent.state_value_estimator is not None:
                value_net_estimates[state_ind] = agent.state_value_estimator(agent, state, target=True).detach().cpu().numpy()
            params['loc'][state_ind, inf_seed] = agent.approx_post.dist.loc.detach().cpu().numpy()[0]
            params['scale'][state_ind, inf_seed] = agent.approx_post.dist.scale.detach().cpu().numpy()[0]
            # grads['loc'][state_ind, inf_seed, ] =
            # grads['scale'][state_ind, inf_seed, ] =

        next_state, reward, done, _ = env.step(action)
        state = env.reset() if done else next_state

    print('Done.')

    analysis_dict = {'states': states,
                     'params': params,
                     'value_estimates': value_estimates}

    if agent_args['approx_post_args']['update'] == 'iterative':
        analysis_dict['it_value_estimates'] = it_value_estimates

    if agent.state_value_estimator is not None:
        analysis_dict['value_net_estimates'] = value_net_estimates

    return analysis_dict


def analyze_inf_training(exp_key, state=None):
    """
    Analyze the approximate posterior distributions throughout training for a
    single state (to check stability during training).

    Args:
        exp_key (str): the string of the comet experiment key
        state (torch.Tensor, optional): state to analyze
    """
    # load the experiment
    comet_api = comet_ml.API(api_key=LOADING_API_KEY)
    experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                          workspace=WORKSPACE,
                                          experiment=exp_key)

    # create the environment
    param_summary = experiment.get_parameters_summary()
    env_name = [a for a in param_summary if a['name'] == 'env'][0]['valueCurrent']
    env = create_env(env_name)

    # create the agent
    asset_list = experiment.get_asset_list()
    agent_config_asset_list = [a for a in asset_list if 'agent_args' in a['fileName']]
    agent_args = None
    if len(agent_config_asset_list) > 0:
        # if we've saved the agent config dict, load it
        agent_args = experiment.get_asset(agent_config_asset_list[0]['assetId'])
        agent_args = json.loads(agent_args)
        agent_args = agent_args if 'opt_type' in agent_args['inference_optimizer_args'] else None
    agent = create_agent(env, agent_args=agent_args)[0]

    # get a state if not provided
    if state is None:
        # load a random state from the most recently collected episode
        asset_times = [asset['createdAt'] for asset in asset_list if 'state' in asset['fileName']]
        state_asset = [a for a in asset_list if a['createdAt'] == max(asset_times)][0]
        episode_states = experiment.get_asset(state_asset['assetId'])
        episode_states = json.loads(episode_states)
        state_timestep = np.random.randint(len(episode_states))
        state = torch.from_numpy(np.array(episode_states[state_timestep])).view(1, -1).type(torch.FloatTensor)

    # get the list of checkpoint timesteps
    ckpt_asset_list = [a for a in asset_list if 'ckpt' in a['fileName']]
    ckpt_asset_names = [a['fileName'] for a in ckpt_asset_list]
    ckpt_timesteps = [int(s.split('ckpt_step_')[1].split('.ckpt')[0]) for s in ckpt_asset_names]

    # to store the approx. post. dists throughout training
    dists = {'loc':[], 'scale':[]}

    # iterate through the checkpoints, estimating approximate posterior
    # iterate over sub-sampled checkpoint timesteps, evaluating
    ckpt_timesteps = list(np.sort(ckpt_timesteps)[::CKPT_SUBSAMPLE])
    for ckpt_ind, ckpt_timestep in enumerate(np.sort(ckpt_timesteps)):
        # load the checkpoint
        print('Evaluating checkpoint ' + str(ckpt_ind + 1) + ' of ' + str(len(ckpt_timesteps)))
        load_checkpoint(agent, exp_key, ckpt_timestep)
        # perform inference
        reward = 0; done = False
        agent.reset(batch_size=1); agent.eval()
        agent.act(state, reward, done)
        dists['loc'].append(agent.approx_post.dist.loc.detach().cpu().numpy()[0])
        dists['scale'].append(agent.approx_post.dist.scale.detach().cpu().numpy()[0])

    return state, ckpt_timesteps, dists

def estimate_amortization_gap(exp_key):
    """
    Estimate the amortization gap for a cached experiment throughout training.

    Args:
        exp_key (str): the string of the comet experiment key
    """
    # number of states to evaluate
    N_STATES = 100

    def estimate_gap(agent, env):
        """
        Sub-routine to estimate amortization gaps.
        """
        gaps = np.zeros(N_STATES)

        agent.reset(); agent.eval()
        state = env.reset()
        reward = 0.
        done = False

        for state_ind in range(N_STATES):
            if state_ind % 10 == 0:
                print(' State ' + str(state_ind) + ' of ' + str(N_STATES) + '.')
            # keep track of objective during inference optimization
            objectives = [np.inf]

            # perform inference using inference model
            action = agent.act(state, reward, done)

            # use gradient-based optimization to estimate amortization gap
            dist_params = {k: v.data.requires_grad_() for k, v in agent.approx_post.get_dist_params().items()}
            agent.approx_post.reset(dist_params=dist_params)
            dist_param_list = [param for _, param in dist_params.items()]
            optimizer = optim.SGD(dist_param_list, lr=1e-4)
            optimizer.zero_grad()
            # initial estimate
            actions = agent.approx_post.sample(agent.n_action_samples)
            obj = agent.estimate_objective(state, actions)
            obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
            objectives.append(obj.detach())

            while objectives[-1] < objectives[-2]:

                obj.sum().backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
                # clear the sample to force resampling
                agent.approx_post._sample = None
                actions = agent.approx_post.sample(agent.n_action_samples)
                obj = agent.estimate_objective(state, actions)
                obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
                objectives.append(obj.detach())

            clear_gradients(agent.generative_parameters())

            gaps[state_ind] = - (objectives[-2] - objectives[1])

            # step the environment
            state, reward, done, _ = env.step(action)
            if done:
                agent.reset(); agent.eval()
                state = env.reset()
                reward = 0.
                done = False

        return gaps

    # load the experiment
    comet_api = comet_ml.API(api_key=LOADING_API_KEY)
    experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                          workspace=WORKSPACE,
                                          experiment=exp_key)

    # create the environment
    param_summary = experiment.get_parameters_summary()
    env_name = [a for a in param_summary if a['name'] == 'env'][0]['valueCurrent']
    env = create_env(env_name)

    # create the agent
    asset_list = experiment.get_asset_list()
    agent_config_asset_list = [a for a in asset_list if 'agent_args' in a['fileName']]
    agent_args = None
    if len(agent_config_asset_list) > 0:
        # if we've saved the agent config dict, load it
        agent_args = experiment.get_asset(agent_config_asset_list[0]['assetId'])
        agent_args = json.loads(agent_args)
        agent_args = agent_args if 'opt_type' in agent_args['inference_optimizer_args'] else None
    agent = create_agent(env, agent_args=agent_args)[0]
    # get the list of checkpoint timesteps
    ckpt_asset_list = [a for a in asset_list if 'ckpt' in a['fileName']]
    ckpt_asset_names = [a['fileName'] for a in ckpt_asset_list]
    ckpt_timesteps = [int(s.split('ckpt_step_')[1].split('.ckpt')[0]) for s in ckpt_asset_names]

    # iterate over sub-sampled checkpoint timesteps, evaluating
    ckpt_timesteps = list(np.sort(ckpt_timesteps)[::CKPT_SUBSAMPLE])
    amortization_gaps = np.zeros((len(ckpt_timesteps), N_STATES))

    for ckpt_ind, ckpt_timestep in enumerate(ckpt_timesteps):
        # load the checkpoint
        print('Evaluating checkpoint ' + str(ckpt_ind + 1) + ' of ' + str(len(ckpt_timesteps)))
        load_checkpoint(agent, exp_key, ckpt_timestep)
        amortization_gaps[ckpt_ind] = estimate_gap(agent, env)

    return ckpt_timesteps, amortization_gaps


def evaluate_optimized_agent(exp_key):
    """
    Evaluate the checkpoints of an experiment using additional gradient-based
    inference optimization steps after amortized inference. Determines the
    degree to which the amortization gap hurts evaluation performance.

    Args:
        exp_key (str): the string of the comet experiment key
    """
    def collect_optimized_episode(env, agent, eval=True):
        """
        Collects an episode of experience using the model and environment. The
        policy distribution is optimized using gradient descent at each step.

        Args:
            env (gym.env): the environment
            agent (Agent): the agent
            random (bool): whether to use random actions
            eval (bool): whether to evaluate the agent

        Returns episode (dict), n_steps (int), and env_states (dict).
        """
        agent.reset(); agent.eval()
        state = env.reset()
        reward = 0.
        done = False
        n_steps = 0
        env_states = {'qpos': [], 'qvel': []}

        n_grad_steps = []
        gaps = []

        while not done:
            if n_steps > 1000:
                break
            if 'sim' in dir(env.unwrapped):
                env_states['qpos'].append(copy.deepcopy(env.sim.data.qpos))
                env_states['qvel'].append(copy.deepcopy(env.sim.data.qvel))
            # perform amortized inference
            agent.act(state, reward, done, None, eval=eval)

            # keep track of objective during inference optimization
            objectives = [np.inf]

            # use gradient-based inference optimization
            dist_params = {k: v.data.requires_grad_() for k, v in agent.approx_post.get_dist_params().items()}
            agent.approx_post.reset(dist_params=dist_params)
            dist_param_list = [param for _, param in dist_params.items()]
            optimizer = optim.SGD(dist_param_list, lr=1e-2)
            optimizer.zero_grad()
            # initial estimate
            actions = agent.approx_post.sample(agent.n_action_samples)
            obj = agent.estimate_objective(state, actions)
            obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
            objectives.append(obj.detach())
            n_g_steps = 0
            while objectives[-1] < objectives[-2] and n_g_steps <= 100:
                dist_params = {k: v.clone() for k, v in agent.approx_post.get_dist_params().items()}
                obj.sum().backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
                # clear the sample to force resampling
                agent.approx_post._sample = None
                actions = agent.approx_post.sample(agent.n_action_samples)
                obj = agent.estimate_objective(state, actions)
                obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
                objectives.append(obj.detach())
                n_g_steps += 1
            clear_gradients(agent.generative_parameters())
            # reset with the second to last set of distribution parameters
            agent.approx_post.reset(dist_params=dist_params)
            agent.approx_post._sample = None
            n_grad_steps.append(n_g_steps)
            gaps.append(- (objectives[-2] - objectives[1]).cpu().numpy().item())

            # sample from the optimized distribution
            action = agent.approx_post.sample(n_samples=1, argmax=eval)
            action = action.tanh() if agent.postprocess_action else action
            action = action.detach().cpu().numpy()

            # step the environment with the optimized action
            state, reward, done, _ = env.step(action)
            n_steps += 1
        if 'sim' in dir(env.unwrapped):
            env_states['qpos'].append(copy.deepcopy(env.sim.data.qpos))
            env_states['qvel'].append(copy.deepcopy(env.sim.data.qvel))
            env_states['qpos'] = np.stack(env_states['qpos'])
            env_states['qvel'] = np.stack(env_states['qvel'])
        print('     Num. Gradient Steps: ' + str(np.mean(n_grad_steps)))
        print('     Average Improvement: ' + str(np.mean(gaps)))
        agent.act(state, reward, done)
        return agent.collector.get_episode(), n_steps, env_states

    # load the experiment
    comet_api = comet_ml.API(api_key=LOADING_API_KEY)
    experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                          workspace=WORKSPACE,
                                          experiment=exp_key)

    # create the environment
    param_summary = experiment.get_parameters_summary()
    env_name = [a for a in param_summary if a['name'] == 'env'][0]['valueCurrent']
    env = create_env(env_name)

    # create the agent
    asset_list = experiment.get_asset_list()
    agent_config_asset_list = [a for a in asset_list if 'agent_args' in a['fileName']]
    agent_args = None
    if len(agent_config_asset_list) > 0:
        # if we've saved the agent config dict, load it
        agent_args = experiment.get_asset(agent_config_asset_list[0]['assetId'])
        agent_args = json.loads(agent_args)
        agent_args = agent_args if 'opt_type' in agent_args['inference_optimizer_args'] else None
    agent = create_agent(env, agent_args=agent_args)[0]
    # get the list of checkpoint timesteps
    ckpt_asset_list = [a for a in asset_list if 'ckpt' in a['fileName']]
    ckpt_asset_names = [a['fileName'] for a in ckpt_asset_list]
    ckpt_timesteps = [int(s.split('ckpt_step_')[1].split('.ckpt')[0]) for s in ckpt_asset_names]

    n_episodes_per_ckpt = 10

    # iterate over sub-sampled checkpoint timesteps, evaluating
    ckpt_timesteps = list(np.sort(ckpt_timesteps)[::CKPT_SUBSAMPLE])
    # amortization_gaps = np.zeros((len(ckpt_timesteps), N_STATES))
    optimized_perf = np.zeros((len(ckpt_timesteps), n_episodes_per_ckpt))
    amortized_perf = np.zeros((len(ckpt_timesteps), n_episodes_per_ckpt))

    for ckpt_ind, ckpt_timestep in enumerate(ckpt_timesteps):
        # load the checkpoint
        print('Evaluating checkpoint ' + str(ckpt_ind + 1) + ' of ' + str(len(ckpt_timesteps)))
        load_checkpoint(agent, exp_key, ckpt_timestep)
        print(' Collecting amortized episodes.')
        for ep_num in range(n_episodes_per_ckpt):
            print(' Ep. ' + str(ep_num))
            amortized_episode, _, _ = collect_episode(env, agent, eval=True)
            amortized_perf[ckpt_ind, ep_num] = amortized_episode['reward'].sum()
        print(' Collecting optimized episodes.')
        for ep_num in range(n_episodes_per_ckpt):
            print(' Ep. ' + str(ep_num))
            optimized_episode, _, _ = collect_optimized_episode(env, agent, eval=True)
            optimized_perf[ckpt_ind, ep_num] = optimized_episode['reward'].sum()

    return {'timesteps': ckpt_timesteps,
            'amortized': amortized_perf,
            'optimized': optimized_perf}

def analyze_1d_inf(exp_key, state=None):
    """
    Calculates the approximate posterior and Boltzmann posterior (both estimated
    and true) for a particular state for the Drone-v0 environment. Helpful to
    visualize the effect of value estimation on inference.

    Args:
        exp_key (str): the string of the comet experiment key
        state (torch.Tensor, optional): state to analyze

    Returns:

    """
    # load the experiment
    comet_api = comet_ml.API(api_key=LOADING_API_KEY)
    experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                          workspace=WORKSPACE,
                                          experiment=exp_key)

    # create the environment
    param_summary = experiment.get_parameters_summary()
    env_name = [a for a in param_summary if a['name'] == 'env'][0]['valueCurrent']
    assert env_name == 'Drone-v0'
    env = create_env(env_name)

    # create the agent
    asset_list = experiment.get_asset_list()
    agent_config_asset_list = [a for a in asset_list if 'agent_args' in a['fileName']]
    agent_args = None
    if len(agent_config_asset_list) > 0:
        # if we've saved the agent config dict, load it
        agent_args = experiment.get_asset(agent_config_asset_list[0]['assetId'])
        agent_args = json.loads(agent_args)
        agent_args = agent_args if 'opt_type' in agent_args['inference_optimizer_args'] else None
    agent = create_agent(env, agent_args=agent_args)[0]

    # load the checkpoint
    load_checkpoint(agent, exp_key)

    # get a state if not provided
    if state is None:
        # load a random state from the most recently collected episode
        asset_times = [asset['createdAt'] for asset in asset_list if '_state' in asset['fileName']]
        state_asset = [a for a in asset_list if a['createdAt'] == max(asset_times)][0]
        episode_states = experiment.get_asset(state_asset['assetId'])
        episode_states = json.loads(episode_states)
        # state_timestep = np.random.randint(len(episode_states))
        state_timestep = 40
        state = torch.from_numpy(np.array(episode_states[state_timestep])).view(1, -1).type(torch.FloatTensor)

    results = {'state': state,
               'value_est': {'amortized_approx_post': None,
                             'optimal_approx_post': None,
                             'boltzmann': None},
               'simulator': {'amortized_approx_post': None,
                             'optimal_approx_post': None,
                             'boltzmann': None}}

    # evaluate actions between -1 and 1 by interval of 0.01
    a = torch.from_numpy(np.arange(-1, 1, 0.01)).view(-1, 1).float()

    # perform amortized inference using the value estimate
    print('Estimating amortized inference dist using value estimate.')
    agent.reset(); agent.eval()
    agent.act(state, 0., False)
    results['value_est']['amortized_approx_post'] = {k: v.clone().detach().numpy() for k, v in agent.approx_post.get_dist_params().items()}

    # get fully-optimized approx post using the value estimate
    print('Estimating optimal inference dist using value estimate.')
    agent.n_action_samples = 1000
    objectives = [np.inf]
    dist_params = {k: v.data.requires_grad_() for k, v in agent.approx_post.get_dist_params().items()}
    agent.approx_post.reset(dist_params=dist_params)
    dist_param_list = [param for _, param in dist_params.items()]
    optimizer = optim.SGD(dist_param_list, lr=1)
    optimizer.zero_grad()
    actions = agent.approx_post.sample(agent.n_action_samples)
    obj = agent.estimate_objective(state, actions)
    obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
    objectives.append(obj.detach())
    # while objectives[-1] < objectives[-2]:
    for _ in range(100):
        obj.sum().backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        agent.approx_post._sample = None
        actions = agent.approx_post.sample(agent.n_action_samples)
        obj = agent.estimate_objective(state, actions)
        obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
        objectives.append(obj.detach())
    clear_gradients(agent.generative_parameters())
    results['value_est']['optimal_approx_post'] = {k: v.clone().detach().numpy() for k, v in agent.approx_post.get_dist_params().items()}
    agent.n_action_samples = 10

    # evaluate the Boltzmann policy using the value estimate
    print('Estimating Boltzmann inference dist using value estimate.')
    expanded_state = state.repeat(a.shape[0], 1)
    q_values = agent.q_value_estimator(agent, expanded_state, a)
    normalizer = agent.alphas['pi'] * ((q_values / agent.alphas['pi']).logsumexp(dim=0, keepdim=True) - torch.tensor(a.shape[0], dtype=torch.float32).log())
    results['value_est']['boltzmann'] = 0.5 * ((q_values - normalizer) / agent.alphas['pi']).exp().detach().numpy()
    results['value_est']['q_values'] = q_values.detach().numpy()

    # perform amortized inference using the simulator
    print('Estimating amortized inference dist using simulator.')
    sim_kwargs = {'estimator_type': 'simulator',
                  'horizon': 250, # int(1. / (1 - agent.reward_discount)),
                  'env_type': env.spec.id}
    # simulator = get_value_estimator('action', sim_kwargs)
    # agent.q_value_estimator = simulator
    agent.reset(); agent.eval()
    agent.act(state, 0., False)
    results['simulator']['amortized_approx_post'] = {k: v.detach().numpy() for k, v in agent.approx_post.get_dist_params().items()}

    # get fully-optimized approx post using the simulator
    print('Estimating optimal inference dist using simulator.')
    # objectives = [np.inf]
    # dist_params = {k: v.data.requires_grad_() for k, v in agent.approx_post.get_dist_params().items()}
    # agent.approx_post.reset(dist_params=dist_params)
    # dist_param_list = [param for _, param in dist_params.items()]
    # optimizer = optim.SGD(dist_param_list, lr=1e-4)
    # optimizer.zero_grad()
    # actions = agent.approx_post.sample(agent.n_action_samples)
    # obj = agent.estimate_objective(state, actions)
    # obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
    # objectives.append(obj.detach())
    # while objectives[-1] < objectives[-2]:
    #     obj.sum().backward(retain_graph=True)
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     agent.approx_post._sample = None
    #     actions = agent.approx_post.sample(agent.n_action_samples)
    #     obj = agent.estimate_objective(state, actions)
    #     obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
    #     objectives.append(obj.detach())
    # clear_gradients(agent.generative_parameters())
    # results['simulator']['optimal_approx_post'] = {k: v.detach().numpy() for k, v in agent.approx_post.get_dist_params().items()}

    # evaluate the Boltzmann policy using the simulator
    print('Estimating Boltzmann inference dist using simulator.')
    # n_samples = 50
    # agent.reset(batch_size=20)
    # q_values = 0
    # for _ in range(n_samples):
    #     q_values = q_values + agent.q_value_estimator(agent, expanded_state, a)
    # q_values = q_values / n_samples
    # normalizer = agent.alphas['pi'] * ((q_values / agent.alphas['pi']).logsumexp(dim=0, keepdim=True) - torch.tensor(a.shape[0], dtype=torch.float32).log())
    # results['simulator']['boltzmann'] = 0.5 * ((q_values - normalizer) / agent.alphas['pi']).exp().detach().numpy()
    # results['simulator']['q_values'] = q_values.detach().numpy()

    ## Test - use regular environment for interaction
    q_values = np.zeros(a.shape)
    # horizon = int(1. / (1 - agent.reward_discount))
    horizon = 1
    for act_ind in range(a.shape[0]):
        print('Action ' + str(act_ind + 1) + ' of ' + str(a.shape[0]) + '.')
        action = a[act_ind:act_ind+1]
        agent.reset(); agent.eval()
        env.reset(); env.model.total_step = 40; env.set_state(state)
        done = False
        total_reward = []
        total_kl = []
        rollout_t = 0
        while not done and rollout_t <= horizon:
            # step the environment
            rollout_state, reward, done, _ = env.step(action)
            total_reward.append(((1 - done) * reward).view(-1).numpy())
            action = agent.act(rollout_state)
            kl = kl_divergence(agent.approx_post, agent.prior, n_samples=agent.n_action_samples).sum(dim=1, keepdim=True)
            total_kl.append(((1 - done) * kl.detach().cpu()).view(-1).numpy())
            rollout_t += 1
        terminal_q = agent.q_value_estimator(agent, rollout_state, torch.from_numpy(action)).detach().cpu().numpy()
        terminal_v = (terminal_q - agent.alphas['pi'].cpu().numpy() * total_kl[-1]).reshape(-1)
        rewards = np.stack(total_reward)
        kls = np.stack(total_kl[:-1])
        discounts = np.cumprod(agent.reward_discount * np.ones(kls.shape), axis=0)
        discounts = np.concatenate([np.ones((1, 1)), discounts]).reshape(-1, 1)
        rewards[1:] = rewards[1:] - agent.alphas['pi'].cpu().numpy() * kls
        sample_return = np.sum(discounts * rewards, axis=0) + (agent.reward_discount ** rollout_t) * terminal_v
        q_values[act_ind, 0] = sample_return
    q_values = torch.from_numpy(q_values)
    normalizer = agent.alphas['pi'] * ((q_values / agent.alphas['pi']).logsumexp(dim=0, keepdim=True) - torch.tensor(a.shape[0], dtype=torch.float32).log())
    results['simulator']['boltzmann'] = 0.5 * ((q_values - normalizer) / agent.alphas['pi']).exp().detach().numpy()
    results['simulator']['q_values'] = q_values.detach().numpy()





    return results
