import comet_ml
import json
import pickle
import numpy as np
from local_vars import PROJECT_NAME, WORKSPACE, LOADING_API_KEY, LOGGING_API_KEY
from lib import create_agent
from util.env_util import create_env
from util.plot_util import load_checkpoint
from util.train_util import collect_episode, estimate_agent_kl
from lib.distributions import kl_divergence

CKPT_SUBSAMPLE = 1
N_STATES = 500

def analyze_agent_kl(exp_key):
    """
    Evaluates the agent KL post-hoc for a given experiment.

    Args:
        exp_key (str): the experiment ID
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

    # get the list of checkpoint timesteps
    ckpt_asset_list = [a for a in asset_list if 'ckpt' in a['fileName']]
    ckpt_asset_names = [a['fileName'] for a in ckpt_asset_list]
    ckpt_timesteps = [int(s.split('ckpt_step_')[1].split('.ckpt')[0]) for s in ckpt_asset_names]
    ckpt_timesteps = list(np.sort(ckpt_timesteps)[::CKPT_SUBSAMPLE])

    agent_kls = []

    # initial episode using random init
    prev_episode, _, _ = collect_episode(env, agent)

    for ckpt_ind, ckpt_timestep in enumerate(ckpt_timesteps):
        # load the checkpoint
        print('Evaluating checkpoint ' + str(ckpt_ind + 1) + ' of ' + str(len(ckpt_timesteps)))
        load_checkpoint(agent, exp_key, ckpt_timestep)

        # evaluate agent KL
        print(' Evaluating agent KL...')
        agent_kls.append(estimate_agent_kl(env, agent, prev_episode))
        print(' Done.')

        # collect an episode
        print(' Collecting episode...')
        prev_episode, _, _ = collect_episode(env, agent)
        print(' Done.')

    return {'steps': ckpt_timesteps, 'agent_kl': np.array(agent_kls)}


def compare_policies(exp_key1, exp_key2, write_result=True):
    """
    Compares the policies of two agents at the end of training.

    Args:
        exp_key1 (str):
        exp_key2 (str):
        write_result (bool)
    """
    # load the experiments
    comet_api = comet_ml.API(api_key=LOADING_API_KEY)
    exp1 = comet_api.get_experiment(project_name=PROJECT_NAME,
                                    workspace=WORKSPACE,
                                    experiment=exp_key1)
    exp2 = comet_api.get_experiment(project_name=PROJECT_NAME,
                                    workspace=WORKSPACE,
                                    experiment=exp_key2)

    # create the environment
    param_summary = exp1.get_parameters_summary()
    env_name = [a for a in param_summary if a['name'] == 'env'][0]['valueCurrent']
    env1 = create_env(env_name)
    env2 = create_env(env_name)

    # create the agents
    asset_list = exp1.get_asset_list()
    agent_config_asset_list = [a for a in asset_list if 'agent_args' in a['fileName']]
    agent_args = None
    if len(agent_config_asset_list) > 0:
        # if we've saved the agent config dict, load it
        agent_args = exp1.get_asset(agent_config_asset_list[0]['assetId'])
        agent_args = json.loads(agent_args)
        agent_args = agent_args if 'opt_type' in agent_args['inference_optimizer_args'] else None
    agent1 = create_agent(env1, agent_args=agent_args)[0]
    load_checkpoint(agent1, exp_key1)

    asset_list = exp2.get_asset_list()
    agent_config_asset_list = [a for a in asset_list if 'agent_args' in a['fileName']]
    agent_args = None
    if len(agent_config_asset_list) > 0:
        # if we've saved the agent config dict, load it
        agent_args = exp2.get_asset(agent_config_asset_list[0]['assetId'])
        agent_args = json.loads(agent_args)
        agent_args = agent_args if 'opt_type' in agent_args['inference_optimizer_args'] else None
    agent2 = create_agent(env1, agent_args=agent_args)[0]
    load_checkpoint(agent2, exp_key2)

    # evaluate the KL between policies
    kl12 = []
    kl21 = []
    agent1.reset(); agent1.eval()
    agent2.reset(); agent2.eval()

    state1 = env1.reset()
    state2 = env2.reset()

    for state_ind in range(N_STATES):
        # perform policy optimization on state1
        action1 = agent1.act(state1)
        agent2.act(state1)
        kl = kl_divergence(agent1.approx_post, agent2.approx_post).sum().detach().item()
        kl12.append(kl)

        agent1.reset(); agent1.eval()
        agent2.reset(); agent2.eval()

        # perform policy optimization on state2
        agent1.act(state2)
        action2 = agent2.act(state2)
        kl = kl_divergence(agent2.approx_post, agent1.approx_post).sum().detach().item()
        kl21.append(kl)

        # step the environments
        state1, _, done1, _ = env1.step(action1)
        state2, _, done2, _ = env2.step(action2)

        if done1:
            agent1.reset(); agent1.eval()
            state1 = env1.reset()
            done1 = False
        if done2:
            agent2.reset(); agent2.eval()
            state2 = env2.reset()
            done2 = False

    kls = {'kl12': kl12,
           'kl21': kl21}

    if write_result:
        pickle.dump(kls, open('policy_kl_' + exp_key1 + '_' + exp_key2 + '.p', 'wb'))

    return kls
