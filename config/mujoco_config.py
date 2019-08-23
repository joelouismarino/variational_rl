import gym.spaces as spaces
import numpy as np
from .get_n_input import get_n_input


def get_mujoco_config(env):
    """
    Get the model configuration arguments for MuJoCo environments.
    """
    agent_args = {}

    agent_args['agent_type'] = 'model_based'

    agent_args['misc_args'] = {'kl_scale': dict(state=1., action=1.),
                               'reward_scale': 1.,
                               'n_state_samples': 2,
                               'n_inf_iter': dict(state=1, action=6),
                               'inference_type': dict(state='direct', action='iterative'),
                               'kl_min': dict(state=0., action=0.),
                               'kl_min_anneal_rate': dict(state=1., action=1.),
                               'kl_factor': dict(state=1., action=1.),
                               'kl_factor_anneal_rate': dict(state=1., action=1.),
                               'reward_discount': 0.99,
                               'normalize_returns': False,
                               'normalize_advantages': False,
                               'normalize_observations': False,
                               'v_trace': dict(l=0.75, iw_clip=1.)}

    if agent_args['agent_type'] == 'generative':
        agent_args['misc_args']['marginal_factor'] = 0.01
        agent_args['misc_args']['marginal_factor_anneal_rate'] = 1.002

    if agent_args['misc_args']['inference_type']['action'] == 'iterative':
        # planning configuration
        agent_args['misc_args']['n_planning_samples'] = 200
        agent_args['misc_args']['max_rollout_length'] = 0

    observation_size = np.prod(env.observation_space.shape)
    agent_args['misc_args']['observation_size'] = observation_size
    action_space = env.action_space
    discrete_actions = False
    if type(action_space) == spaces.Discrete:
        # discrete control
        # TODO: used reparameterized categorical?
        action_prior_dist = 'Categorical'
        action_approx_post_dist = 'Categorical'
        # n_action_variables = env.action_space.n
        n_action_variables = 3
        discrete_actions = True
    elif type(action_space) == spaces.Box:
        # continuous control
        if env.action_space.low.min() == -1 and env.action_space.high.max() == 1:
            # action_prior_dist = 'TransformedTanh'
            action_prior_dist = 'TransformedTanh'
            action_approx_post_dist = 'TransformedTanh'
        else:
            action_prior_dist = 'Normal'
            action_approx_post_dist = 'Normal'
        n_action_variables = env.action_space.shape[0]
        discrete_actions = False
    else:
        raise NotImplementedError

    if agent_args['agent_type'] == 'baseline':
        # action
        agent_args['action_variable_args'] = {'type': 'fully_connected',
                                              'prior_dist': action_prior_dist,
                                              'approx_post_dist': action_approx_post_dist,
                                              'n_variables': n_action_variables,
                                              'constant_prior': True,
                                              'inference_type': 'direct'}

        # agent_args['action_prior_args'] = {'type': 'fully_connected',
        #                                    'n_layers': 2,
        #                                    'inputs': ['observation'],
        #                                    'n_units': 256,
        #                                    'connectivity': 'sequential',
        #                                    'batch_norm': False,
        #                                    'non_linearity': 'relu',
        #                                    'dropout': None}

        agent_args['action_prior_args'] = None

        agent_args['action_inference_args'] = {'type': 'fully_connected',
                                               'n_layers': 2,
                                               'inputs': ['observation'],
                                               'n_units': 256,
                                               'connectivity': 'sequential',
                                               'batch_norm': False,
                                               'non_linearity': 'relu',
                                               'dropout': None}

        agent_args['q_value_model_args'] = {'type': 'fully_connected',
                                          'n_layers': 2,
                                          'inputs': ['observation', 'action'],
                                          'n_units': 256,
                                          'connectivity': 'sequential',
                                          'non_linearity': 'relu',
                                          'dropout': None}

    elif agent_args['agent_type'] == 'model_based':

        # action
        agent_args['action_variable_args'] = {'type': 'fully_connected',
                                              'prior_dist': action_prior_dist,
                                              'approx_post_dist': action_approx_post_dist,
                                              'n_variables': n_action_variables,
                                              'constant_prior': False,
                                              'inference_type': 'iterative'}

        agent_args['action_prior_args'] = {'type': 'fully_connected',
                                           'n_layers': 2,
                                           'inputs': ['observation'],
                                           'n_units': 256,
                                           'connectivity': 'sequential',
                                           'batch_norm': False,
                                           'non_linearity': 'relu',
                                           'dropout': None}

        agent_args['action_inference_args'] = {'type': 'fully_connected',
                                               'n_layers': 2,
                                               'inputs': ['params', 'grads'],
                                               'n_units': 256,
                                               'connectivity': 'sequential',
                                               'batch_norm': False,
                                               'non_linearity': 'relu',
                                               'dropout': None}

        # agent_args['action_inference_args'] = None

        # observation (state)
        agent_args['observation_variable_args'] = {'type': 'fully_connected',
                                                   'likelihood_dist': 'Normal',
                                                   'n_variables': observation_size,
                                                   'constant_scale': True,
                                                   'sigmoid_loc': False}

        agent_args['obs_likelihood_args'] = {'type': 'fully_connected',
                                             'n_layers': 2,
                                             'inputs': ['observation', 'action'],
                                             'n_units': 256,
                                             'connectivity': 'sequential',
                                             'batch_norm': False,
                                             'non_linearity': 'relu'}

        # reward
        agent_args['reward_variable_args'] = {'type': 'fully_connected',
                                              'likelihood_dist': 'Normal',
                                              'n_variables': 1,
                                              'constant_scale': True,
                                              'sigmoid_loc': False}

        agent_args['reward_likelihood_args'] = {'type': 'fully_connected',
                                                'n_layers': 2,
                                                'inputs': ['observation', 'action'],
                                                'n_units': 256,
                                                'connectivity': 'sequential',
                                                'batch_norm': False,
                                                'non_linearity': 'relu',
                                                'dropout': None}

        agent_args['q_value_model_args'] = {'type': 'fully_connected',
                                          'n_layers': 2,
                                          'inputs': ['observation', 'action'],
                                          'n_units': 256,
                                          'connectivity': 'sequential',
                                          'non_linearity': 'relu',
                                          'dropout': None}

    if agent_args['agent_type'] == 'discriminative':
        # state
        n_state_variables = 64
        agent_args['state_variable_args'] = {'type': 'fully_connected',
                                             'prior_dist': 'Normal',
                                             'approx_post_dist': 'Normal',
                                             'n_variables': n_state_variables,
                                             'inference_type': 'direct'}

        agent_args['state_prior_args'] = {'type': 'fully_connected',
                                          'n_layers': 1,
                                          'inputs': ['observation'],
                                          'n_units': 128,
                                          'connectivity': 'highway',
                                          'non_linearity': 'elu',
                                          'dropout': None}

        agent_args['state_inference_args'] = {'type': 'fully_connected',
                                              'n_layers': 1,
                                              'inputs': ['observation'],
                                              'n_units': 128,
                                              'connectivity': 'highway',
                                              'non_linearity': 'elu',
                                              'dropout': None}

        # action
        agent_args['action_variable_args'] = {'type': 'fully_connected',
                                              'prior_dist': action_prior_dist,
                                              'approx_post_dist': action_approx_post_dist,
                                              'n_variables': n_action_variables,
                                              'constant_prior': False,
                                              'inference_type': 'direct'}

        agent_args['action_prior_args'] = {'type': 'fully_connected',
                                           'n_layers': 2,
                                           'inputs': ['state'],
                                           'n_units': 64,
                                           'connectivity': 'highway',
                                           'batch_norm': False,
                                           'non_linearity': 'elu',
                                           'dropout': None}

        # agent_args['action_prior_args'] = None

        agent_args['action_inference_args'] = {'type': 'fully_connected',
                                               'n_layers': 2,
                                               'inputs': ['state'],
                                               'n_units': 64,
                                               'connectivity': 'highway',
                                               'batch_norm': False,
                                               'non_linearity': 'elu',
                                               'dropout': None}

        agent_args['value_model_args'] = {'type': 'fully_connected',
                                          'n_layers': 2,
                                          'inputs': ['state'],
                                          'n_units': 64,
                                          'connectivity': 'highway',
                                          'non_linearity': 'tanh',
                                          'dropout': None}

        agent_args['q_value_model_args'] = {'type': 'fully_connected',
                                            'n_layers': 2,
                                            'inputs': ['state', 'action'],
                                            'n_units': 64,
                                            'connectivity': 'highway',
                                            'non_linearity': 'relu',
                                            'dropout': None}

    if agent_args['agent_type'] == 'generative':
        # state
        n_state_variables = 128
        agent_args['state_variable_args'] = {'type': 'fully_connected',
                                             'prior_dist': 'Normal',
                                             'approx_post_dist': 'Normal',
                                             'n_variables': n_state_variables,
                                             'norm_samples': True,
                                             'inference_type': agent_args['misc_args']['inference_type']['state']}

        agent_args['state_prior_args'] = {'type': 'recurrent',
                                          'n_layers': 1,
                                          'inputs': ['state', 'action'],
                                          'n_units': 128,
                                          'connectivity': 'highway',
                                          'dropout': None}

        # hidden_state_size = agent_args['state_prior_args']['n_layers'] * agent_args['state_prior_args']['n_units']

        agent_args['state_inference_args'] = {'type': 'fully_connected',
                                              'n_layers': 1,
                                              'inputs': ['params', 'grads'],
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
                                              'constant_prior': False,
                                              'inference_type': agent_args['misc_args']['inference_type']['action']}

        if agent_args['action_variable_args']['inference_type'] == 'iterative':
            # model-based action inference
            # agent_args['action_prior_args'] = None
            agent_args['action_prior_args'] = {'type': 'fully_connected',
                                               'n_layers': 2,
                                               'inputs': ['state', 'action'],
                                               'n_units': 64,
                                               'connectivity': 'highway',
                                               'batch_norm': False,
                                               'non_linearity': 'elu',
                                               'dropout': None}

            agent_args['action_inference_args'] = {'type': 'fully_connected',
                                                   'n_layers': 1,
                                                   'inputs': ['params', 'grads'],
                                                   'n_units': 256,
                                                   'connectivity': 'highway',
                                                   'batch_norm': False,
                                                   'non_linearity': 'elu',
                                                   'dropout': None}
        else:
            # model-free action inference
            agent_args['action_prior_args'] = {'type': 'fully_connected',
                                               'n_layers': 2,
                                               'inputs': ['state', 'action'],
                                               'n_units': 64,
                                               'connectivity': 'highway',
                                               'batch_norm': False,
                                               'non_linearity': 'elu',
                                               'dropout': None}

            agent_args['action_inference_args'] = {'type': 'fully_connected',
                                                   'n_layers': 2,
                                                   'inputs': ['state', 'action'],
                                                   'n_units': 64,
                                                   'connectivity': 'highway',
                                                   'batch_norm': False,
                                                   'non_linearity': 'elu',
                                                   'dropout': None}

        # observation
        agent_args['observation_variable_args'] = {'type': 'fully_connected',
                                                   'likelihood_dist': 'Normal',
                                                   'n_variables': observation_size,
                                                   'constant_scale': True,
                                                   'sigmoid_loc': False}

        agent_args['obs_likelihood_args'] = {'type': 'fully_connected',
                                             'n_layers': 1,
                                             'inputs': ['state'],
                                             'n_units': 128,
                                             'connectivity': 'highway',
                                             'batch_norm': True,
                                             'non_linearity': 'elu'}

        # reward
        agent_args['reward_variable_args'] = {'type': 'fully_connected',
                                              'likelihood_dist': 'Normal',
                                              'n_variables': 1,
                                              'constant_scale': True,
                                              'sigmoid_loc': False}

        agent_args['reward_likelihood_args'] = {'type': 'fully_connected',
                                                'n_layers': 1,
                                                'inputs': ['state'],
                                                'n_units': 64,
                                                'connectivity': 'highway',
                                                'batch_norm': True,
                                                'non_linearity': 'elu',
                                                'dropout': None}

        # done
        agent_args['done_variable_args'] = {'type': 'fully_connected',
                                            'likelihood_dist': 'Bernoulli',
                                            'n_variables': 1}

        agent_args['done_likelihood_args'] = {'type': 'fully_connected',
                                              'n_layers': 1,
                                              'inputs': ['state'],
                                              'n_units': 64,
                                              'connectivity': 'sequential',
                                              'batch_norm': True,
                                              'non_linearity': 'tanh',
                                              'dropout': None}

        # value
        agent_args['value_model_args'] = {'type': 'fully_connected',
                                          'n_layers': 2,
                                          'inputs': ['state'],
                                          'n_units': 64,
                                          'connectivity': 'highway',
                                          'non_linearity': 'tanh',
                                          'dropout': None}

        agent_args['q_value_model_args'] = {'type': 'fully_connected',
                                            'n_layers': 2,
                                            'inputs': ['state', 'action'],
                                            'n_units': 64,
                                            'connectivity': 'highway',
                                            'non_linearity': 'relu',
                                            'dropout': None}

    # calculate the input sizes for all models
    agent_args = get_n_input(agent_args, discrete_actions=discrete_actions)
    return agent_args
