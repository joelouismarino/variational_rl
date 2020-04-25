import comet_ml
import json
import numpy as np
from lib import create_agent
from util.env_util import create_env
from util.plot_util import load_checkpoint
from local_vars import PROJECT_NAME, WORKSPACE, LOADING_API_KEY, LOGGING_API_KEY

# load a checkpoint
# collect one or more episodes
# visualize inference over multiple seeds
# analyze inference performance (improvement, gap?)


def analyze_inference(exp_key, n_states, n_inf_seeds, n_action_samples=None,
                      ckpt_timestep=None, device_id=None):
    """
    Analyzes the inference procedure of a cached experiment.

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
    inf_iter_list = [a for a in param_summary if a['name'] == 'inference_optimizer_args_n_inf_iters']
    n_inf_iters = 1
    if len(inf_iter_list) > 0:
        n_inf_iters = int(inf_iter_list[0]['valueCurrent'])
    n_state_dims = env.observation_space.shape[-1]
    n_action_dims = env.action_space.shape[0]

    states = np.zeros((n_states, n_state_dims))
    # actions = np.array((n_states, n_inf_seeds, n_action_samples, n_action_dims))
    value_estimates = np.zeros((n_states, n_inf_seeds, n_inf_iters, 1))
    params = {'loc': np.zeros((n_states, n_inf_seeds, n_action_dims)),
              'scale': np.zeros((n_states, n_inf_seeds, n_action_dims))}
    # grads = {'loc': np.array((n_states, n_inf_seeds, n_action_dims)),
    #          'scale': np.array((n_states, n_inf_seeds, n_action_dims))}
    print('Collecting ' + str(n_states) + ' states...')
    state = env.reset(); reward = 0; done = False
    for state_ind in range(n_states):
        states[state_ind] = state[0]

        for inf_seed in range(n_inf_seeds):
            agent.reset(batch_size=1); agent.eval()
            action = agent.act(state, reward, done)
            # save the quantities
            # actions[state_ind, inf_seed, ] = action
            q_ests = [obj.detach().cpu().numpy() for obj in agent.inference_optimizer.estimated_objectives]
            value_estimates[state_ind, inf_seed] = np.stack(q_ests).reshape(-1, 1)
            params['loc'][state_ind, inf_seed] = agent.approx_post.dist.loc.detach().cpu().numpy()[0]
            params['scale'][state_ind, inf_seed] = agent.approx_post.dist.scale.detach().cpu().numpy()[0]
            # grads['loc'][state_ind, inf_seed, ] =
            # grads['scale'][state_ind, inf_seed, ] =

        next_state, reward, done, _ = env.step(action)
        state = env.reset() if done else next_state

    print('Done.')

    return {'states': states,
            'params': params}
