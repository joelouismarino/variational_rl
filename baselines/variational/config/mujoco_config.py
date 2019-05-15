import gym.spaces as spaces
import numpy as np


def get_mujoco_config(env):
    """
    Get the model configuration arguments for MuJoCo environments.
    """
    agent_args = {}

    agent_args['agent_type'] = 'generative'

    agent_args['misc_args'] = {'optimality_scale': 1,
                               'n_state_samples': 1,
                               'n_inf_iter': dict(state=1, action=2),
                               'kl_min': dict(state=0., action=0.),
                               'kl_min_anneal_rate': dict(state=1., action=1.),
                               'kl_factor': dict(state=1., action=1.),
                               'kl_factor_anneal_rate': dict(state=1., action=1.004),
                               'reward_discount': 0.99,
                               'normalize_returns': True,
                               'normalize_advantages': True,
                               'normalize_observations': False,
                               'gae_lambda': 0.98}

    if agent_args['misc_args']['n_inf_iter']['action'] > 0:
        # planning configuration
        agent_args['misc_args']['n_planning_samples'] = 200
        agent_args['misc_args']['max_rollout_length'] = 10

    observation_size = np.prod(env.observation_space.shape)
    action_space = env.action_space
    if type(action_space) == spaces.Discrete:
        # discrete control
        # TODO: used reparameterized categorical?
        action_prior_dist = 'Categorical'
        action_approx_post_dist = 'Categorical'
        # n_action_variables = env.action_space.n
        n_action_variables = 3
        action_inf_n_input = 2 * n_action_variables # not currently used
    elif type(action_space) == spaces.Box:
        # continuous control
        action_prior_dist = 'Normal'
        action_approx_post_dist = 'Normal'
        n_action_variables = env.action_space.shape[0]
        action_inf_n_input = 4 * n_action_variables
    else:
        raise NotImplementedError


    if agent_args['agent_type'] == 'discriminative':
        # state
        n_state_variables = 64
        agent_args['state_variable_args'] = {'type': 'fully_connected',
                                             'prior_dist': 'Delta', #'Normal' or 'Delta'
                                             'approx_post_dist': None,
                                             'n_variables': n_state_variables}

        agent_args['state_prior_args'] = {'type': 'fully_connected',
                                          'n_layers': 1,
                                          'n_input': observation_size,
                                          'n_units': 64,
                                          'connectivity': 'sequential',
                                          'non_linearity': 'tanh',
                                          'dropout': None}
        # agent_args['state_prior_args'] = {'type': 'minigrid_conv'}
        # hidden_state_size = agent_args['state_prior_args']['n_layers'] * agent_args['state_prior_args']['n_units']

        agent_args['state_inference_args'] = None

        # action
        agent_args['action_variable_args'] = {'type': 'fully_connected',
                                              'prior_dist': action_prior_dist,
                                              'approx_post_dist': action_approx_post_dist,
                                              'n_variables': n_action_variables,
                                              'constant_prior': False,
                                              'inference_type': 'direct'}

        agent_args['action_prior_args'] = {'type': 'fully_connected',
                                               'n_layers': 1,
                                               'n_input': n_state_variables,
                                               'n_units': 64,
                                               'connectivity': 'sequential',
                                               'batch_norm': False,
                                               'non_linearity': 'tanh',
                                               'dropout': None}

        # agent_args['action_prior_args'] = None

        agent_args['action_inference_args'] = {'type': 'fully_connected',
                                               'n_layers': 1,
                                               'n_input': n_state_variables,
                                               'n_units': 64,
                                               'connectivity': 'sequential',
                                               'batch_norm': False,
                                               'non_linearity': 'tanh',
                                               'dropout': None}

        agent_args['value_model_args'] = {'type': 'fully_connected',
                                          'n_layers': 2,
                                          'n_input': n_state_variables,
                                          'n_units': 64,
                                          'connectivity': 'highway',
                                          'non_linearity': 'elu',
                                          'dropout': None}

    if agent_args['agent_type'] == 'generative':
        # state
        n_state_variables = 128
        agent_args['state_variable_args'] = {'type': 'fully_connected',
                                             'prior_dist': 'Normal',
                                             'approx_post_dist': 'Normal',
                                             'n_variables': n_state_variables,
                                             'norm_samples': True,
                                             'inference_type': 'iterative'}

        agent_args['state_prior_args'] = {'type': 'recurrent',
                                          'n_layers': 1,
                                          'n_input': n_state_variables + n_action_variables,
                                          'n_units': 128,
                                          'connectivity': 'highway',
                                          'dropout': None}

        # hidden_state_size = agent_args['state_prior_args']['n_layers'] * agent_args['state_prior_args']['n_units']

        agent_args['state_inference_args'] = {'type': 'fully_connected',
                                              'n_layers': 1,
                                              'n_input': 4 * n_state_variables,
                                              'n_units': 256,
                                              'connectivity': 'highway',
                                              'batch_norm': False,
                                              'non_linearity': 'elu',
                                              'dropout': None}

        # action
        agent_args['action_variable_args'] = {'type': 'fully_connected',
                                              'prior_dist': action_prior_dist,
                                              'approx_post_dist': action_approx_post_dist,
                                              'n_variables': n_action_variables,
                                              'constant_prior': True,
                                              'inference_type': 'iterative'}

        if agent_args['action_variable_args']['inference_type'] == 'iterative':
            # model-based action inference
            agent_args['action_prior_args'] = None
            # agent_args['action_prior_args'] = {'type': 'fully_connected',
            #                                    'n_layers': 2,
            #                                    'n_input': n_state_variables + n_action_variables,
            #                                    'n_units': 64,
            #                                    'connectivity': 'highway',
            #                                    'batch_norm': False,
            #                                    'non_linearity': 'elu',
            #                                    'dropout': None}

            agent_args['action_inference_args'] = {'type': 'fully_connected',
                                                   'n_layers': 1,
                                                   'n_input': 4 * n_action_variables,
                                                   'n_units': 128,
                                                   'connectivity': 'highway',
                                                   'batch_norm': False,
                                                   'non_linearity': 'elu',
                                                   'dropout': None}
        else:
            # model-free action inference
            agent_args['action_prior_args'] = {'type': 'fully_connected',
                                               'n_layers': 2,
                                               'n_input': n_state_variables + n_action_variables,
                                               'n_units': 64,
                                               'connectivity': 'highway',
                                               'batch_norm': False,
                                               'non_linearity': 'elu',
                                               'dropout': None}

            agent_args['action_inference_args'] = {'type': 'fully_connected',
                                                   'n_layers': 2,
                                                   'n_input': n_state_variables + n_action_variables,
                                                   'n_units': 64,
                                                   'connectivity': 'highway',
                                                   'batch_norm': False,
                                                   'non_linearity': 'elu',
                                                   'dropout': None}

        # observation
        agent_args['observation_variable_args'] = {'type': 'fully_connected',
                                              'likelihood_dist': 'Normal',
                                              'n_variables': observation_size}

        agent_args['obs_likelihood_args'] = {'type': 'fully_connected',
                                             'n_layers': 2,
                                             'n_input': n_state_variables,
                                             'n_units': 128,
                                             'connectivity': 'highway',
                                             'non_linearity': 'elu'}

        # reward
        agent_args['reward_variable_args'] = {'type': 'fully_connected',
                                              'likelihood_dist': 'Normal',
                                              'integration_window': None,
                                              'n_variables': 1,
                                              'sigmoid_loc': False}

        agent_args['reward_likelihood_args'] = {'type': 'fully_connected',
                                                'n_layers': 2,
                                                'n_input': n_state_variables,
                                                'n_units': 64,
                                                'connectivity': 'highway',
                                                'batch_norm': False,
                                                'non_linearity': 'elu',
                                                'dropout': None}

        # done
        agent_args['done_variable_args'] = {'type': 'fully_connected',
                                            'likelihood_dist': 'Bernoulli',
                                            'n_variables': 1}

        agent_args['done_likelihood_args'] = {'type': 'fully_connected',
                                              'n_layers': 1,
                                              'n_input': n_state_variables,
                                              'n_units': 64,
                                              'connectivity': 'sequential',
                                              'batch_norm': False,
                                              'non_linearity': 'tanh',
                                              'dropout': None}

        # value
        agent_args['value_model_args'] = {'type': 'fully_connected',
                                          'n_layers': 2,
                                          'n_input': n_state_variables,
                                          'n_units': 64,
                                          'connectivity': 'highway',
                                          'non_linearity': 'elu',
                                          'dropout': None}

    return agent_args
