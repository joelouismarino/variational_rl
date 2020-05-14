import comet_ml, json
import numpy as np
import torch
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
