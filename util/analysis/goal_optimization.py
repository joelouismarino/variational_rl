import comet_ml
import json
import torch
import copy
import pickle
import numpy as np
from torch import optim
from lib import create_agent
from lib.layers import FullyConnectedLayer
from lib.value_estimators import GoalBasedQEstimator
from lib.inference import GradientBasedInference, IterativeInferenceModel
from util.env_util import create_env, SynchronousEnv
from util.plot_util import load_checkpoint
from local_vars import PROJECT_NAME, WORKSPACE, LOADING_API_KEY, LOGGING_API_KEY
from misc import divide_gradients_by_value

N_TOTAL_STEPS = 200
GOAL_INTERVAL = 20
GOAL_STD = 1e-2
RENDER = False
BATCH_SIZE = 256
GOAL_FLIP_PROB = 0.05
ENCODING_TYPE = 'grads'

TRAJECTORY_FOLLOW = False

# iterative optimizer exp: b1c6214082ed4e74af70c4215e275406
# model-based exp: 3bf28c960c224b989d26fa7c1237dd5e

def goal_optimization(model_exp_key, opt_exp_key=None, write_results=True):
    """
    Optimize random goal states using a model-based estimator.
    Note: tailored to HalfCheetah-v2 environment currently.

    Args:
        model_exp_key (str): model-based experiment key
        opt_exp_key (str): optimizer experiment key
        write_results (bool): whether to pickle results directly
    """

    # load the experiment
    comet_api = comet_ml.API(api_key=LOADING_API_KEY)
    experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                          workspace=WORKSPACE,
                                          experiment=model_exp_key)

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

    # also, load the most recent episode to sample goal states
    asset_times = [asset['createdAt'] for asset in asset_list if 'state' in asset['fileName']]
    state_asset = [a for a in asset_list if a['createdAt'] == max(asset_times)][0]
    episode_states = json.loads(experiment.get_asset(state_asset['assetId']))

    # load the checkpoint
    load_checkpoint(agent, model_exp_key)

    # load the optimizer
    if opt_exp_key is not None:
        # load the experiment
        comet_api = comet_ml.API(api_key=LOADING_API_KEY)
        opt_experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                                  workspace=WORKSPACE,
                                                  experiment=opt_exp_key)

        # create the agent
        asset_list = opt_experiment.get_asset_list()
        agent_config_asset_list = [a for a in asset_list if 'agent_args' in a['fileName']]
        agent_args = None
        if len(agent_config_asset_list) > 0:
            # if we've saved the agent config dict, load it
            agent_args = opt_experiment.get_asset(agent_config_asset_list[0]['assetId'])
            agent_args = json.loads(agent_args)
            agent_args = agent_args if 'opt_type' in agent_args['inference_optimizer_args'] else None
        opt_agent = create_agent(env, agent_args=agent_args)[0]

        # load the checkpoint
        load_checkpoint(opt_agent, opt_exp_key)

        agent.inference_optimizer = opt_agent.inference_optimizer
        agent.inference_optimizer.n_inf_iters = 20
    else:
        # create a gradient-based optimizer
        agent.inference_optimizer = GradientBasedInference(lr=1e-3, n_inf_iters=50)

    # swap out the value estimator for goal-based estimator
    gb_estimator = GoalBasedQEstimator()
    # copy over the dynamics model
    gb_estimator.state_likelihood_model = agent.q_value_estimator.state_likelihood_model
    gb_estimator.state_variable = agent.q_value_estimator.state_variable
    # set the estimator
    agent.q_value_estimator = gb_estimator
    agent.q_value_estimator.set_goal_std(GOAL_STD)
    # agent.alphas['pi'] = 0.

    # optimize goal states
    goal_states = []
    traj_states = []
    env_states = {'qpos': [], 'qvel': []}
    actions = []
    inf_objectives = []

    agent.reset(); agent.eval()
    state = env.reset()
    if RENDER:
        env.render()
    goal_state = None

    # goal_ind = 0

    print('Collecting goal-optimization episode...')
    for step_ind in range(N_TOTAL_STEPS):
        print('STEP: ' + str(step_ind))
        if step_ind % GOAL_INTERVAL == 0:
            goal_state = episode_states[np.random.randint(0, 25)]
            # goal_state = episode_states[goal_ind]
            goal_state = torch.from_numpy(np.array(goal_state)).float().view(1,-1)
            goal_state[:, 8:] *= 0.
            if not TRAJECTORY_FOLLOW:
                agent.q_value_estimator.set_goal_state(goal_state)
            # goal_ind += 1
        if TRAJECTORY_FOLLOW:
            # define a sub-goal between current state and goal state
            delta_state = goal_state - state
            traj_state = state + 0.1 * delta_state
            agent.q_value_estimator.set_goal_state(traj_state)
            traj_states.append(traj_state)
        else:
            traj_states.append(goal_states)
        goal_states.append(goal_state)
        env_states['qpos'].append(copy.deepcopy(env.sim.data.qpos))
        env_states['qvel'].append(copy.deepcopy(env.sim.data.qvel))
        action = agent.act(state, eval=True)
        state, _, _, _ = env.step(action)
        inf_objectives.append(agent.inference_optimizer.estimated_objectives)
        # import ipdb; ipdb.set_trace()
        agent.inference_optimizer.reset(1)
        if RENDER:
            env.render()
        actions.append(action)
    print('Done.')

    # save the results
    results = {'goal_states': goal_states,
               'traj_states': traj_states,
               'env_states': env_states,
               'actions': actions}

    if write_results:
        pickle.dump(results, open('goal_opt_' + model_exp_key + '.p', 'wb'))

    return results


def goal_optimization_training(model_exp_key, opt_exp_key=None, write_results=True, stochastic_model=False):
    """
    Optimize random goal states using a model-based estimator.
    Train the policy optimizer online.
    Note: tailored to HalfCheetah-v2 environment currently.

    Args:
        model_exp_key (str): model-based experiment key
        opt_exp_key (str): optimizer experiment key. If None, trains from scratch
        write_results (bool): whether to pickle results directly
    """

    # load the experiment
    comet_api = comet_ml.API(api_key=LOADING_API_KEY)
    experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                          workspace=WORKSPACE,
                                          experiment=model_exp_key)

    # create the environment
    param_summary = experiment.get_parameters_summary()
    env_name = [a for a in param_summary if a['name'] == 'env'][0]['valueCurrent']
    env = create_env(env_name)
    # create a synchronous env to parallelize training
    sync_env = SynchronousEnv(env, BATCH_SIZE)

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

    # also, load the most recent episode to sample goal states
    asset_times = [asset['createdAt'] for asset in asset_list if 'state' in asset['fileName']]
    state_asset = [a for a in asset_list if a['createdAt'] == max(asset_times)][0]
    episode_states = json.loads(experiment.get_asset(state_asset['assetId']))

    # load the checkpoint
    load_checkpoint(agent, model_exp_key)

    if stochastic_model:
        agent.q_value_estimator.state_variable.cond_likelihood.stochastic = True

    # load the optimizer
    if opt_exp_key is not None:
        # load the experiment
        comet_api = comet_ml.API(api_key=LOADING_API_KEY)
        opt_experiment = comet_api.get_experiment(project_name=PROJECT_NAME,
                                                  workspace=WORKSPACE,
                                                  experiment=opt_exp_key)

        # create the agent
        asset_list = opt_experiment.get_asset_list()
        agent_config_asset_list = [a for a in asset_list if 'agent_args' in a['fileName']]
        agent_args = None
        if len(agent_config_asset_list) > 0:
            # if we've saved the agent config dict, load it
            agent_args = opt_experiment.get_asset(agent_config_asset_list[0]['assetId'])
            agent_args = json.loads(agent_args)
            agent_args = agent_args if 'opt_type' in agent_args['inference_optimizer_args'] else None
        opt_agent = create_agent(env, agent_args=agent_args)[0]

        # load the checkpoint
        load_checkpoint(opt_agent, opt_exp_key)

        agent.inference_optimizer = opt_agent.inference_optimizer
        agent.inference_optimizer.n_inf_iters = 10
    else:
        # create an iterative amortized optimizer
        n_input = 12
        if ENCODING_TYPE == 'grads':
            inputs = ['params', 'grads']
            n_input += 12
        elif ENCODING_TYPE == 'errors':
            inputs = ['params', 'errors']
            n_input += (17 + 17 + 6)
        n_units = 512
        # network_args = {'type': 'fully_connected',
        #                 'n_layers': 2,
        #                 'inputs': inputs,
        #                 'n_units': n_units,
        #                 'connectivity': 'highway',
        #                 'batch_norm': False,
        #                 'non_linearity': 'elu',
        #                 'dropout': None,
        #                 'separate_networks': False,
        #                 'n_input': n_input}
        network_args = {'type': 'recurrent',
                        'n_layers': 2,
                        'inputs': inputs,
                        'n_units': n_units,
                        'connectivity': 'highway',
                        'batch_norm': False,
                        'dropout': None,
                        'separate_networks': False,
                        'n_input': n_input}
        agent.inference_optimizer = IterativeInferenceModel(network_args=network_args, n_inf_iters=5, encoding_type=ENCODING_TYPE)
        for m in agent.approx_post.models:
            agent.approx_post.models[m] = FullyConnectedLayer(n_units, 6)
            agent.approx_post.gates[m] = FullyConnectedLayer(n_units, 6, non_linearity='sigmoid')

    # create a parameter optimizer for the inference model
    inference_parameters = [_ for _ in agent.inference_optimizer.parameters()] + [_ for _ in agent.approx_post.parameters()]
    param_opt = optim.Adam(inference_parameters, lr=3e-4)

    # swap out the value estimator for goal-based estimator
    gb_estimator = GoalBasedQEstimator()
    # copy over the dynamics model
    gb_estimator.state_likelihood_model = agent.q_value_estimator.state_likelihood_model
    gb_estimator.state_variable = agent.q_value_estimator.state_variable
    # set the estimator
    agent.q_value_estimator = gb_estimator
    agent.q_value_estimator.set_goal_std(GOAL_STD)
    # agent.alphas['pi'] = 0.

    # optimize goal states
    goal_states = []
    traj_states = []
    env_states = {'qpos': [], 'qvel': []}
    actions = []
    inf_objectives = []

    agent.reset(batch_size=BATCH_SIZE); agent.eval()
    state = sync_env.reset()
    if RENDER:
        env.render()
    goal_state = None

    # goal_ind = 0

    print('Collecting goal-optimization episode...')
    for step_ind in range(N_TOTAL_STEPS):
        print('STEP: ' + str(step_ind))
        # if step_ind % GOAL_INTERVAL == 0:
        if True:
            new_goal_states = np.stack([episode_states[np.random.randint(0, 25)] for _ in range(BATCH_SIZE)])
            # goal_state = episode_states[goal_ind]
            new_goal_states = torch.from_numpy(new_goal_states).float().view(BATCH_SIZE,-1)
            new_goal_states[:, 8:] *= 0.
            if step_ind == 0:
                goal_state = new_goal_states
            else:
                # randomly change the goal state with some small probability
                flips = (torch.rand(BATCH_SIZE, 1) < GOAL_FLIP_PROB).float().repeat(1, new_goal_states.shape[-1])
                goal_state = (1 - flips) * goal_state + flips * new_goal_states
            if not TRAJECTORY_FOLLOW:
                agent.q_value_estimator.set_goal_state(goal_state)
            # goal_ind += 1
        if TRAJECTORY_FOLLOW:
            # define a sub-goal between current state and goal state
            delta_state = goal_state - state
            traj_state = state + 0.1 * delta_state
            agent.q_value_estimator.set_goal_state(traj_state)
            traj_states.append(traj_state)
        else:
            traj_states.append(goal_states)
        goal_states.append(goal_state)
        qpos = np.stack([copy.deepcopy(e.sim.data.qpos) for e in sync_env.envs])
        qvel = np.stack([copy.deepcopy(e.sim.data.qvel) for e in sync_env.envs])
        env_states['qpos'].append(qpos)
        env_states['qvel'].append(qvel)
        action = agent.act(state, eval=True)
        state, _, _, _ = sync_env.step(action)
        inf_objectives.append(agent.inference_optimizer.estimated_objectives)

        # update the inference optimizer
        grads = [param.grad for param in inference_parameters]
        divide_gradients_by_value(grads, agent.inference_optimizer.n_inf_iters)
        divide_gradients_by_value(grads, BATCH_SIZE)
        param_opt.step()
        param_opt.zero_grad()

        agent.inference_optimizer.reset(BATCH_SIZE)
        if RENDER:
            env.render()
        actions.append(action)
    print('Done.')

    # save the results
    results = {'goal_states': goal_states,
               'traj_states': traj_states,
               'env_states': env_states,
               'actions': actions}

    if write_results:
        pickle.dump(results, open('goal_opt_' + model_exp_key + '.p', 'wb'))

    return results
