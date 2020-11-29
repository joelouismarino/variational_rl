import comet_ml
import json
import torch
import copy
import pickle
import numpy as np
from lib import create_agent
from lib.value_estimators import GoalBasedQEstimator
from util.env_util import create_env
from util.plot_util import load_checkpoint
from local_vars import PROJECT_NAME, WORKSPACE, LOADING_API_KEY, LOGGING_API_KEY

N_TOTAL_STEPS = 500
GOAL_INTERVAL = 50
GOAL_STD = 1e-2

def goal_optimization(exp_key, write_results=True):
    """
    Optimize random goal states using a model-based estimator.
    Note: tailored to HalfCheetah-v2 environment currently.

    Args:
        exp_key (str): (model-based) experiment key
        write_results (bool): whether to pickle results directly
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

    # also, load the most recent episode to sample goal states
    asset_times = [asset['createdAt'] for asset in asset_list if 'state' in asset['fileName']]
    state_asset = [a for a in asset_list if a['createdAt'] == max(asset_times)][0]
    episode_states = json.loads(experiment.get_asset(state_asset['assetId']))

    # load the checkpoint
    load_checkpoint(agent, exp_key)

    # swap out the value estimator for goal-based estimator
    gb_estimator = GoalBasedQEstimator()
    # copy over the dynamics model
    gb_estimator.state_likelihood_model = agent.q_value_estimator.state_likelihood_model
    gb_estimator.state_variable = agent.q_value_estimator.state_variable
    # set the estimator
    agent.q_value_estimator = gb_estimator
    agent.q_value_estimator.set_goal_std(GOAL_STD)
    agent.alphas['pi'] = 0.01

    # optimize goal states
    goal_states = []
    env_states = {'qpos': [], 'qvel': []}
    actions = []

    agent.reset(); agent.eval()
    state = env.reset()
    env.render()
    goal_state = None

    # goal_ind = 0

    print('Collecting goal-optimization episode...')
    for step_ind in range(N_TOTAL_STEPS):
        if step_ind % GOAL_INTERVAL == 0:
            goal_state = episode_states[np.random.randint(0, 25)]
            # goal_state = episode_states[goal_ind]
            goal_state = torch.from_numpy(np.array(goal_state)).float().view(1,-1)
            agent.q_value_estimator.set_goal_state(goal_state)
            # goal_ind += 1
        goal_states.append(goal_state)
        env_states['qpos'].append(copy.deepcopy(env.sim.data.qpos))
        env_states['qvel'].append(copy.deepcopy(env.sim.data.qvel))
        action = agent.act(state, eval=True)
        state, _, _, _ = env.step(action)
        env.render()
        actions.append(action)
    print('Done.')

    # save the results
    results = {'goal_states': goal_states,
               'env_states': env_states,
               'actions': actions}

    if write_results:
        pickle.dump(results, open('goal_opt_' + exp_key + '.p', 'wb'))

    return results
