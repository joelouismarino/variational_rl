import comet_ml, json
import numpy as np
import torch
from torch import optim
from misc import clear_gradients
from lib import create_agent
from util.env_util import create_env
from util.plot_util import load_checkpoint
from lib.distributions import kl_divergence
from local_vars import PROJECT_NAME, WORKSPACE, LOADING_API_KEY, LOGGING_API_KEY

alim = [-1, 1]
aint = 0.01

BATCH_SIZE = 256
N_ACTION_SAMPLES = 100

# Reacher Experiments:
# direct: 48edb0b9aca847c09c6893793c982884
# iterative: 58ec5bc5273044e59ae30a969c3d7de4

def estimate_opt_landscape(exp_key, states=None, ckpt_timestep=None, device_id=None):
    """
    Estimates the optimization landscape for a checkpointed agent. Also gets the
    policy estimates during inference optimization.

    Args:
        exp_key (str): the comet experiment ID
        state (list of torch.Tensor, optional): the state(s) used for estimation
        ckpt_timestep (int, optional): the checkpoint for estimation
        device_id (int, optional): the GPU ID
    """
    # load the experiment
    comet_api = comet_ml.API(api_key=LOADING_API_KEY)
    experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                          workspace=WORKSPACE,
                                          experiment=exp_key)

    # create the environment (just to create agent)
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

    if states is None:
        # load a random state from the most recently collected episode
        state_asset = None
        if ckpt_timestep is not None:
            # load the corresponding episode if it is present
            state_asset_list = [a for a in asset_list if 'episode_step_' + str(ckpt_timestep) + '_state' in a['fileName']]
            if len(state_asset_list) > 0:
                state_asset = state_asset_list[0]
        if state_asset is None:
            # get most recently collected episode
            asset_times = [asset['createdAt'] for asset in asset_list if 'state' in asset['fileName']]
            state_asset = [a for a in asset_list if a['createdAt'] == max(asset_times)][0]

        episode_states = experiment.get_asset(state_asset['assetId'])
        episode_states = json.loads(episode_states)
        # state_timestep = [np.random.randint(len(episode_states)) for _ in range(100)]
        state_timesteps = range(100)
        states = [torch.from_numpy(np.array(episode_states[state_timestep])).view(1, -1).type(torch.FloatTensor) for state_timestep in state_timesteps]

    # n_actions = int(((alim[1] - alim[0]) / aint) ** n_action_dims)
    n_action_dims = env.action_space.shape[0]
    a = np.arange(alim[0], alim[1], aint)
    a_args = n_action_dims * [a]
    a_coords = np.meshgrid(*a_args)
    stacked_action_means = np.stack([a_coord.reshape(-1) for a_coord in a_coords]).T
    n_batches = len(stacked_action_means) // BATCH_SIZE + 1
    n_samples = agent.n_action_samples if N_ACTION_SAMPLES is None else N_ACTION_SAMPLES

    q_estimates_list = []
    log_ratios_list = []
    approx_posts_list = []

    for state_ind, state in enumerate(states):
        if state_ind % 5 == 0:
            print('Processing state ' + str(state_ind+1) + ' of ' +  str(len(states)) + '.')
        q_estimates = np.zeros(len(stacked_action_means))
        log_ratios = np.zeros(len(stacked_action_means))

        # perform inference on the state
        batch_expanded_state = state.repeat(BATCH_SIZE, 1)
        sample_expanded_state = batch_expanded_state.repeat(n_samples, 1)
        agent.reset(batch_size=BATCH_SIZE); agent.eval()
        agent.act(batch_expanded_state)
        approx_posts = agent.inference_optimizer.dist_params

        # loop over actions, get value estimates
        for batch_ind in range(n_batches):
            if batch_ind % 25 == 0:
                print(' Processing batch ' + str(batch_ind+1) + ' of ' +  str(n_batches) + '.')
            # get a batch of actions
            start_ind = batch_ind * BATCH_SIZE
            end_ind = min((batch_ind + 1) * BATCH_SIZE, len(stacked_action_means))
            action_mean_batch = stacked_action_means[start_ind:end_ind]
            # evaluate the value estimate of the action in the state
            if action_mean_batch.shape[0] != BATCH_SIZE:
                # fill out the rest of the batch with zeros
                temp_action_mean_batch = np.zeros((BATCH_SIZE, n_action_dims))
                temp_action_mean_batch[:action_mean_batch.shape[0]] = action_mean_batch
                action_mean_batch = temp_action_mean_batch

            action_mean_batch = torch.from_numpy(action_mean_batch).type(torch.FloatTensor)

            # reset approx post, sample actions
            agent.reset(batch_size=BATCH_SIZE); agent.eval()
            agent.approx_post.reset(batch_size=BATCH_SIZE, dist_params={'loc': action_mean_batch.clone().requires_grad_()})
            # agent.inference_optimizer(agent, batch_expanded_state)
            action_batch = agent.approx_post.sample(n_samples)

            q_values = agent.q_value_estimator(agent, sample_expanded_state, action_batch)
            q_values = q_values.view(n_samples, -1, 1).mean(dim=0)
            kls = kl_divergence(agent.approx_post, agent.prior, n_samples=n_samples, sample=action_batch).sum(dim=1, keepdim=True)

            q_estimates[start_ind:end_ind] = q_values[:end_ind-start_ind].view(-1).detach().cpu().numpy()
            log_ratios[start_ind:end_ind] = kls[:end_ind-start_ind].view(-1).detach().cpu().numpy()

        q_estimates = q_estimates.reshape(n_action_dims * [int((alim[1] - alim[0]) / aint)])
        log_ratios = log_ratios.reshape(n_action_dims * [int((alim[1] - alim[0]) / aint)])

        q_estimates_list.append(q_estimates)
        log_ratios_list.append(log_ratios)
        approx_posts_list.append(approx_posts)

    return {'q_estimates': q_estimates_list,
            'log_ratios': log_ratios_list,
            'alpha_pi': agent.alphas['pi'].detach().cpu().numpy(),
            'approx_posts': approx_posts_list}


def vis_inference(exp_key, action_indices, state_ind=0):
    """
    Plots a 2D analysis of direct inference, comparing with gradient ascent.

    Args:
        exp_key (str): the experiment key
        state_ind (int): state index to plot
        action_indices (list): two action indices to vary
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

    # load the checkpoint
    load_checkpoint(agent, exp_key)

    # load the state from the most recently collected episode
    asset_times = [asset['createdAt'] for asset in asset_list if 'state' in asset['fileName']]
    state_asset = [a for a in asset_list if a['createdAt'] == max(asset_times)][0]
    episode_states = json.loads(experiment.get_asset(state_asset['assetId']))
    state = torch.from_numpy(np.array(episode_states[state_ind])).view(1, -1).type(torch.FloatTensor)
    print('STATE: ')
    print(state)

    # perform inference, get the direct approx post
    agent.reset(); agent.eval()
    agent.act(state)
    loc = agent.approx_post.dist.loc.detach().clone(); scale = agent.approx_post.dist.scale.detach().clone()
    direct_approx_post = {'loc': loc.clone().cpu().numpy(),
                          'scale': scale.clone().cpu().numpy()}
    print('DIRECT APPROX. POST.: ')
    print(direct_approx_post)

    print('Performing gradient-based optimization...')
    LR = 0.1
    agent.n_action_samples = 100
    # update the approx post using gradient descent on the 2 dims of the mean
    sgd_objectives = [np.inf]
    sgd_locs = [agent.approx_post.dist.loc.detach().cpu().numpy()]
    # dist_params = {k: v.data.requires_grad_() for k, v in agent.approx_post.get_dist_params().items()}
    sgd_loc = agent.approx_post.dist.loc.clone().detach().requires_grad_()
    sgd_scale = agent.approx_post.dist.scale.clone().detach()
    dist_params = {'loc': sgd_loc, 'scale': sgd_scale}
    agent.approx_post.reset(dist_params=dist_params)
    # dist_param_list = [param for _, param in dist_params.items()]
    # just perform SGD on the mean
    dist_param_list = [sgd_loc]
    # optimizer = optim.SGD(dist_param_list, lr=LR, momentum=0.9)
    optimizer = optim.Adam(dist_param_list, lr=LR)
    optimizer.zero_grad()
    actions = agent.approx_post.sample(agent.n_action_samples)
    obj = agent.estimate_objective(state, actions)
    obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
    sgd_objectives.append(-obj.detach())
    for _ in range(250):
        obj.sum().backward(retain_graph=True)
        for a_dim in range(agent.approx_post.dist.loc.shape[1]):
            if a_dim not in action_indices:
                agent.approx_post.dist.loc.grad[:, a_dim] = 0.
        optimizer.step()
        optimizer.zero_grad()
        agent.approx_post._sample = None
        # reset the non-optimized dimensions
        # for a_dim in range(agent.approx_post.dist.loc.shape[1]):
        #     if a_dim not in action_indices:
        #         agent.approx_post.dist.loc[:, a_dim] = loc[:, a_dim]
        # agent.approx_post.dist.scale = scale
        sgd_locs.append(agent.approx_post.dist.loc.clone().detach().cpu().numpy())

        # reset the optimizer, pretty hacky...
        # sgd_loc = agent.approx_post.dist.loc.clone().detach().requires_grad_()
        # sgd_scale = agent.approx_post.dist.scale.clone().detach()
        # dist_params = {'loc': sgd_loc, 'scale': sgd_scale}
        # agent.approx_post.reset(dist_params=dist_params)
        # # dist_param_list = [param for _, param in dist_params.items()]
        # # just perform SGD on the mean
        # dist_param_list = [sgd_loc]
        # optimizer = optim.Adam(dist_param_list, lr=LR)
        # optimizer.zero_grad()

        actions = agent.approx_post.sample(agent.n_action_samples)
        obj = agent.estimate_objective(state, actions)
        obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
        sgd_objectives.append(-obj.detach())
    clear_gradients(agent.generative_parameters())
    agent.n_action_samples = 10
    print('Done.')

    print('Estimating objectives...')
    agent.n_action_samples = 10

    # get all action means
    a = np.arange(alim[0], alim[1], aint)
    a_args = 2 * [a]
    a_coords = np.meshgrid(*a_args)
    stacked_action_means = np.stack([a_coord.reshape(-1) for a_coord in a_coords]).T
    n_batches = len(stacked_action_means) // BATCH_SIZE + 1
    n_samples = agent.n_action_samples
    batch_expanded_state = state.repeat(BATCH_SIZE, 1)
    batch_expanded_loc = loc.repeat(BATCH_SIZE, 1)
    batch_expanded_scale = scale.repeat(BATCH_SIZE, 1)

    objectives = np.zeros((len(stacked_action_means), 1))

    # estimate the objective at all action means
    for batch_ind in range(n_batches):
        if batch_ind % 25 == 0:
            print(' Processing batch ' + str(batch_ind+1) + ' of ' +  str(n_batches) + '.')
        # get a batch of actions
        start_ind = batch_ind * BATCH_SIZE
        end_ind = min((batch_ind + 1) * BATCH_SIZE, len(stacked_action_means))
        action_mean_batch = stacked_action_means[start_ind:end_ind]
        if action_mean_batch.shape[0] != BATCH_SIZE:
            # fill out the rest of the batch with zeros if at the end
            temp_action_mean_batch = np.zeros((BATCH_SIZE, 2))
            temp_action_mean_batch[:action_mean_batch.shape[0]] = action_mean_batch
            action_mean_batch = temp_action_mean_batch
        action_mean_batch = torch.from_numpy(np.arctanh(action_mean_batch + 1e-6)).type(torch.FloatTensor)

        # reset approx post, sample actions
        agent.reset(batch_size=BATCH_SIZE); agent.eval()
        loc_batch = batch_expanded_loc
        loc_batch[:, action_indices[0]] = action_mean_batch[:, 0]
        loc_batch[:, action_indices[1]] = action_mean_batch[:, 1]
        scale_batch = batch_expanded_scale
        agent.approx_post.reset(batch_size=BATCH_SIZE, dist_params={'loc': loc_batch.clone().requires_grad_(),
                                                                    'scale': scale_batch.clone()})
        action_batch = agent.approx_post.sample(n_samples)

        # evaluate the value estimate of the action in the state
        objective = agent.estimate_objective(batch_expanded_state, action_batch).view(n_samples, -1, 1).mean(dim=0).detach().cpu().numpy()
        objectives[start_ind:end_ind] = objective[:end_ind-start_ind]

    objectives = objectives.reshape(2 * [int((alim[1] - alim[0]) / aint)])
    agent.n_action_samples = 10

    print('Done.')

    return {'objectives': objectives,
            'stacked_action_means': stacked_action_means,
            'direct_approx_post': direct_approx_post,
            'action_indices': action_indices,
            'sgd_approx_post_means': sgd_locs,
            'sgd_objectives': sgd_objectives}


def vis_it_inference(exp_key, action_indices, state_ind=0):
    """
    Plots a 2D analysis of iterative inference.

    Args:
        exp_key (str): the experiment key
        state_ind (int): state index to plot
        action_indices (list): two action indices to vary
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

    # load the checkpoint
    load_checkpoint(agent, exp_key)

    # load the state from the most recently collected episode
    asset_times = [asset['createdAt'] for asset in asset_list if 'state' in asset['fileName']]
    state_asset = [a for a in asset_list if a['createdAt'] == max(asset_times)][0]
    episode_states = json.loads(experiment.get_asset(state_asset['assetId']))
    state = torch.from_numpy(np.array(episode_states[state_ind])).view(1, -1).type(torch.FloatTensor)
    print('STATE: ')
    print(state)

    # perform iterative inference, get the approx post
    agent.reset(); agent.eval()
    agent.act(state)
    loc = agent.approx_post.dist.loc.detach().clone(); scale = agent.approx_post.dist.scale.detach().clone()
    it_approx_post = {'loc': loc.clone().cpu().numpy(),
                      'scale': scale.clone().cpu().numpy()}
    print('ITERATIVE APPROX. POST.: ')
    print(it_approx_post)

    print('Performing iterative inference...')
    # only optimize two of the means
    agent.n_action_samples = 100
    agent.reset(); agent.eval()
    iterative_locs = []
    iterative_objectives = []
    # for inf_it in range(agent.inference_optimizer.n_inf_iters):
    for inf_it in range(25):

        # reset the approx post dist
        it_loc = agent.approx_post.dist.loc.clone().detach()
        for a_dim in range(it_loc.shape[1]):
            if a_dim not in action_indices:
                it_loc[:, a_dim] = loc[:, a_dim]
        it_scale = scale
        dist_params = {'loc': it_loc.requires_grad_(), 'scale': it_scale.requires_grad_()}
        agent.approx_post.reset(dist_params=dist_params)
        iterative_locs.append(agent.approx_post.dist.loc.detach().cpu().numpy())

        # estimate the objective, backprop
        actions = agent.approx_post.sample(agent.n_action_samples)
        obj = agent.estimate_objective(state, actions)
        obj = - obj.view(agent.n_action_samples, -1, 1).mean(dim=0)
        iterative_objectives.append(-obj.detach())
        obj.sum().backward(retain_graph=True)

        # update
        params, grads = agent.approx_post.params_and_grads()
        inf_input = agent.inference_optimizer.inference_model(params=params, grads=grads, state=state)
        agent.approx_post.step(inf_input)
        agent.approx_post.retain_grads()

    # reset the approx post dist
    it_loc = agent.approx_post.dist.loc.clone().detach()
    for a_dim in range(it_loc.shape[1]):
        if a_dim not in action_indices:
            it_loc[:, a_dim] = loc[:, a_dim]
    it_scale = scale
    dist_params = {'loc': it_loc, 'scale': it_scale}
    agent.approx_post.reset(dist_params=dist_params)
    iterative_locs.append(agent.approx_post.dist.loc.detach().cpu().numpy())
    print('Done.')

    print('Estimating objectives...')
    agent.n_action_samples = 10

    # get all action means
    a = np.arange(alim[0], alim[1], aint)
    a_args = 2 * [a]
    a_coords = np.meshgrid(*a_args)
    stacked_action_means = np.stack([a_coord.reshape(-1) for a_coord in a_coords]).T
    n_batches = len(stacked_action_means) // BATCH_SIZE + 1
    n_samples = agent.n_action_samples
    batch_expanded_state = state.repeat(BATCH_SIZE, 1)
    batch_expanded_loc = loc.repeat(BATCH_SIZE, 1)
    batch_expanded_scale = scale.repeat(BATCH_SIZE, 1)

    objectives = np.zeros((len(stacked_action_means), 1))

    # estimate the objective at all action means
    for batch_ind in range(n_batches):
        if batch_ind % 25 == 0:
            print(' Processing batch ' + str(batch_ind+1) + ' of ' +  str(n_batches) + '.')
        # get a batch of actions
        start_ind = batch_ind * BATCH_SIZE
        end_ind = min((batch_ind + 1) * BATCH_SIZE, len(stacked_action_means))
        action_mean_batch = stacked_action_means[start_ind:end_ind]
        if action_mean_batch.shape[0] != BATCH_SIZE:
            # fill out the rest of the batch with zeros if at the end
            temp_action_mean_batch = np.zeros((BATCH_SIZE, 2))
            temp_action_mean_batch[:action_mean_batch.shape[0]] = action_mean_batch
            action_mean_batch = temp_action_mean_batch
        action_mean_batch = torch.from_numpy(np.arctanh(action_mean_batch + 1e-6)).type(torch.FloatTensor)

        # reset approx post, sample actions
        agent.reset(batch_size=BATCH_SIZE); agent.eval()
        loc_batch = batch_expanded_loc
        loc_batch[:, action_indices[0]] = action_mean_batch[:, 0]
        loc_batch[:, action_indices[1]] = action_mean_batch[:, 1]
        scale_batch = batch_expanded_scale
        agent.approx_post.reset(batch_size=BATCH_SIZE, dist_params={'loc': loc_batch.clone().requires_grad_(),
                                                                    'scale': scale_batch.clone()})
        action_batch = agent.approx_post.sample(n_samples)

        # evaluate the value estimate of the action in the state
        objective = agent.estimate_objective(batch_expanded_state, action_batch).view(n_samples, -1, 1).mean(dim=0).detach().cpu().numpy()
        objectives[start_ind:end_ind] = objective[:end_ind-start_ind]

    objectives = objectives.reshape(2 * [int((alim[1] - alim[0]) / aint)])
    agent.n_action_samples = 10

    print('Done.')

    return {'objectives': objectives,
            'stacked_action_means': stacked_action_means,
            'action_indices': action_indices,
            'iterative_approx_post_means': iterative_locs,
            'iterative_objectives': iterative_objectives,
            'final_it_approx_post': it_approx_post,}


def compare_inference(direct_exp_key, iterative_exp_key, action_indices, state_ind=0):
    """
    Plots a 2D analysis of direct inference, comparing with an iterative inference agent.

    Args:
        direct_exp_key (str): the experiment key for the direct agent
        iterative_exp_key (str): the experiment key for the iterative agent
        action_indices (list): two action indices to vary
        state_ind (int): state index to plot
    """
    # load the experiment
    comet_api = comet_ml.API(api_key=LOADING_API_KEY)
    direct_experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                                 workspace=WORKSPACE,
                                                 experiment=direct_exp_key)

    iterative_experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                                    workspace=WORKSPACE,
                                                    experiment=iterative_exp_key)

    # create the environment
    dir_param_summary = direct_experiment.get_parameters_summary()
    dir_env_name = [a for a in dir_param_summary if a['name'] == 'env'][0]['valueCurrent']
    it_param_summary = iterative_experiment.get_parameters_summary()
    it_env_name = [a for a in it_param_summary if a['name'] == 'env'][0]['valueCurrent']
    assert it_env_name == dir_env_name
    env = create_env(dir_env_name)

    # create the agents
    # direct
    dir_asset_list = direct_experiment.get_asset_list()
    dir_agent_config_asset_list = [a for a in dir_asset_list if 'agent_args' in a['fileName']]
    dir_agent_args = None
    if len(dir_agent_config_asset_list) > 0:
        # if we've saved the agent config dict, load it
        dir_agent_args = direct_experiment.get_asset(dir_agent_config_asset_list[0]['assetId'])
        dir_agent_args = json.loads(dir_agent_args)
        dir_agent_args = dir_agent_args if 'opt_type' in dir_agent_args['inference_optimizer_args'] else None
    dir_agent = create_agent(env, agent_args=dir_agent_args)[0]

    # iterative
    it_asset_list = iterative_experiment.get_asset_list()
    it_agent_config_asset_list = [a for a in it_asset_list if 'agent_args' in a['fileName']]
    it_agent_args = None
    if len(it_agent_config_asset_list) > 0:
        # if we've saved the agent config dict, load it
        it_agent_args = iterative_experiment.get_asset(it_agent_config_asset_list[0]['assetId'])
        it_agent_args = json.loads(it_agent_args)
        it_agent_args = it_agent_args if 'opt_type' in it_agent_args['inference_optimizer_args'] else None
    it_agent = create_agent(env, agent_args=it_agent_args)[0]

    # load the checkpoints
    load_checkpoint(dir_agent, direct_exp_key)
    load_checkpoint(it_agent, iterative_exp_key)

    # load the state from the most recently collected episode
    asset_times = [asset['createdAt'] for asset in dir_asset_list if 'state' in asset['fileName']]
    state_asset = [a for a in dir_asset_list if a['createdAt'] == max(asset_times)][0]
    episode_states = json.loads(direct_experiment.get_asset(state_asset['assetId']))
    state = torch.from_numpy(np.array(episode_states[state_ind])).view(1, -1).type(torch.FloatTensor)
    print('STATE: ')
    print(state)

    # perform inference, get the direct approx post
    print('Performing direct inference...')
    dir_agent.reset(); dir_agent.eval()
    dir_agent.act(state)
    dir_loc = dir_agent.approx_post.dist.loc.detach().clone(); dir_scale = dir_agent.approx_post.dist.scale.detach().clone()
    direct_approx_post = {'loc': dir_loc.clone().cpu().numpy(),
                          'scale': dir_scale.clone().cpu().numpy()}
    print('DIRECT APPROX. POST.: ')
    print(direct_approx_post)
    print('Done.')

    print('Performing iterative inference...')
    it_agent.n_action_samples = 100
    it_agent.q_value_estimator = dir_agent.q_value_estimator
    it_agent.reset(); it_agent.eval()
    iterative_locs = []
    iterative_objectives = []
    # for inf_it in range(it_agent.inference_optimizer.n_inf_iters):
    for inf_it in range(25):

        # reset the approx post dist
        it_loc = it_agent.approx_post.dist.loc.clone().detach()
        for a_dim in range(it_loc.shape[1]):
            if a_dim not in action_indices:
                it_loc[:, a_dim] = dir_loc[:, a_dim]
        it_scale = dir_scale
        dist_params = {'loc': it_loc.requires_grad_(), 'scale': it_scale.requires_grad_()}
        it_agent.approx_post.reset(dist_params=dist_params)
        iterative_locs.append(it_agent.approx_post.dist.loc.detach().cpu().numpy())

        # estimate the objective, backprop
        actions = it_agent.approx_post.sample(it_agent.n_action_samples)
        obj = it_agent.estimate_objective(state, actions)
        obj = - obj.view(it_agent.n_action_samples, -1, 1).mean(dim=0)
        iterative_objectives.append(-obj.detach())
        obj.sum().backward(retain_graph=True)

        # update
        params, grads = it_agent.approx_post.params_and_grads()
        inf_input = it_agent.inference_optimizer.inference_model(params=params, grads=grads, state=state)
        it_agent.approx_post.step(inf_input)
        it_agent.approx_post.retain_grads()

    # reset the approx post dist
    it_loc = it_agent.approx_post.dist.loc.clone().detach()
    for a_dim in range(it_loc.shape[1]):
        if a_dim not in action_indices:
            it_loc[:, a_dim] = dir_loc[:, a_dim]
    it_scale = dir_scale
    dist_params = {'loc': it_loc, 'scale': it_scale}
    it_agent.approx_post.reset(dist_params=dist_params)
    iterative_locs.append(it_agent.approx_post.dist.loc.detach().cpu().numpy())
    print('Done.')

    print('Estimating objectives...')
    dir_agent.n_action_samples = 10

    # get all action means
    a = np.arange(alim[0], alim[1], aint)
    a_args = 2 * [a]
    a_coords = np.meshgrid(*a_args)
    stacked_action_means = np.stack([a_coord.reshape(-1) for a_coord in a_coords]).T
    n_batches = len(stacked_action_means) // BATCH_SIZE + 1
    n_samples = dir_agent.n_action_samples
    batch_expanded_state = state.repeat(BATCH_SIZE, 1)
    batch_expanded_loc = dir_loc.repeat(BATCH_SIZE, 1)
    batch_expanded_scale = dir_scale.repeat(BATCH_SIZE, 1)

    objectives = np.zeros((len(stacked_action_means), 1))

    # estimate the objective at all action means
    for batch_ind in range(n_batches):
        if batch_ind % 25 == 0:
            print(' Processing batch ' + str(batch_ind+1) + ' of ' +  str(n_batches) + '.')
        # get a batch of actions
        start_ind = batch_ind * BATCH_SIZE
        end_ind = min((batch_ind + 1) * BATCH_SIZE, len(stacked_action_means))
        action_mean_batch = stacked_action_means[start_ind:end_ind]
        if action_mean_batch.shape[0] != BATCH_SIZE:
            # fill out the rest of the batch with zeros if at the end
            temp_action_mean_batch = np.zeros((BATCH_SIZE, 2))
            temp_action_mean_batch[:action_mean_batch.shape[0]] = action_mean_batch
            action_mean_batch = temp_action_mean_batch
        action_mean_batch = torch.from_numpy(np.arctanh(action_mean_batch + 1e-6)).type(torch.FloatTensor)

        # reset approx post, sample actions
        dir_agent.reset(batch_size=BATCH_SIZE); dir_agent.eval()
        loc_batch = batch_expanded_loc
        loc_batch[:, action_indices[0]] = action_mean_batch[:, 0]
        loc_batch[:, action_indices[1]] = action_mean_batch[:, 1]
        scale_batch = batch_expanded_scale
        dir_agent.approx_post.reset(batch_size=BATCH_SIZE, dist_params={'loc': loc_batch.clone().requires_grad_(),
                                                                        'scale': scale_batch.clone()})
        action_batch = dir_agent.approx_post.sample(n_samples)

        # evaluate the value estimate of the action in the state
        objective = dir_agent.estimate_objective(batch_expanded_state, action_batch).view(n_samples, -1, 1).mean(dim=0).detach().cpu().numpy()
        objectives[start_ind:end_ind] = objective[:end_ind-start_ind]

    objectives = objectives.reshape(2 * [int((alim[1] - alim[0]) / aint)])
    dir_agent.n_action_samples = 10

    print('Done.')

    return {'objectives': objectives,
            'stacked_action_means': stacked_action_means,
            'direct_approx_post': direct_approx_post,
            'action_indices': action_indices,
            'iterative_approx_post_means': iterative_locs,
            'iterative_objectives': iterative_objectives}


def vis_mb_opt(exp_key, state_ind, rollout_horizon=None):
    """
    Evaluates iterative amortized policy optimization on model-based value
    estimates.

    Args:
        exp_key (str): the experiment key for the (model-based) agent
        state_ind (int): state index to evaluate
        rollout_horizon (int): the MB rollout horizon (None for agent default)

    Returns dictionary containing:
        states (np.array) [n iters, horizon, n action samples, state dim]
        rewards (np.array)
        q_values (np.array)
        actions (np.array)
        objectives (np.array)
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

    # set the rollout horizon to a different length
    if rollout_horizon is not None:
        agent.q_value_estimator.horizon = rollout_horizon

    # load the checkpoint
    load_checkpoint(agent, exp_key)

    # load the state from the most recently collected episode
    asset_times = [asset['createdAt'] for asset in asset_list if 'state' in asset['fileName'] and '.svg' not in asset['fileName']]
    state_asset = [a for a in asset_list if a['createdAt'] == max(asset_times)][0]
    episode_states = json.loads(experiment.get_asset(state_asset['assetId']))
    state = torch.from_numpy(np.array(episode_states[state_ind])).view(1, -1).type(torch.FloatTensor)
    print('STATE: ')
    print(state)

    # perform iterative inference, get the approx post
    agent.reset(); agent.eval()
    agent.act(state)

    rollout_states = agent.q_value_estimator.rollout_states
    rollout_rewards = agent.q_value_estimator.rollout_rewards
    rollout_q_values = agent.q_value_estimator.rollout_q_values
    rollout_actions = agent.q_value_estimator.rollout_actions

    states = torch.stack([torch.stack(rs, dim=0).detach().cpu() for rs in rollout_states], dim=0).numpy()
    rewards = torch.stack([torch.stack(rr, dim=0).detach().cpu() for rr in rollout_rewards], dim=0).numpy()
    q_values = torch.stack([torch.stack(rq, dim=0).detach().cpu() for rq in rollout_q_values], dim=0).numpy()
    actions = torch.stack([torch.stack(ra, dim=0).detach().cpu() for ra in rollout_actions], dim=0).numpy()

    objectives = agent.inference_optimizer.estimated_objectives
    final_actions = agent.approx_post.sample(agent.n_action_samples)
    final_obj = agent.estimate_objective(state, final_actions)
    objectives.append( - final_obj.view(agent.n_action_samples, -1, 1).mean(dim=0))
    objs = np.array([-obj.item() for obj in objectives])


    return {'states': states,
            'rewards': rewards,
            'q_values': q_values,
            'action': actions,
            'objectives': objs}
