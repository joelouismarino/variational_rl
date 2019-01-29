import gym.spaces as spaces
import numpy as np


def get_minigrid_config(env):
    """
    Get the model configuration arguments for MiniGrid environments.
    """
    model_args = {}

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

    # state
    n_state_variables = 100
    model_args['state_variable_args'] = {'type': 'fully_connected',
                                         'prior_dist': 'Normal',
                                         'approx_post_dist': 'Normal',
                                         'n_variables': n_state_variables}

    model_args['state_prior_args'] = {'type': 'recurrent',
                                      'n_layers': 1,
                                      'n_input': n_state_variables + n_action_variables,
                                      'n_units': 512,
                                      'connectivity': 'sequential',
                                      'dropout': None}

    hidden_state_size = model_args['state_prior_args']['n_layers'] * model_args['state_prior_args']['n_units']

    model_args['state_inference_args'] = {'type': 'fully_connected',
                                          'n_layers': 1,
                                          'n_input': 4 * n_state_variables,
                                          'n_units': 1024,
                                          'connectivity': 'highway',
                                          'batch_norm': False,
                                          'non_linearity': 'elu',
                                          'dropout': None}

    # action
    model_args['action_variable_args'] = {'type': 'fully_connected',
                                          'prior_dist': action_prior_dist,
                                          'approx_post_dist': action_approx_post_dist,
                                          'n_variables': n_action_variables}

    model_args['action_prior_args'] = {'type': 'fully_connected',
                                       'n_layers': 1,
                                       'n_input': n_state_variables + n_action_variables,
                                       'n_units': 200,
                                       'connectivity': 'sequential',
                                       'batch_norm': False,
                                       'non_linearity': 'relu',
                                       'dropout': None}

    model_args['action_inference_args'] = {'type': 'fully_connected',
                                           'n_layers': 1,
                                           'n_input': n_state_variables + n_action_variables,
                                           'n_units': 200,
                                           'connectivity': 'sequential',
                                           'batch_norm': False,
                                           'non_linearity': 'relu',
                                           'dropout': None}

    # observation
    model_args['observation_variable_args'] = {'type': 'fully_connected',
                                               'likelihood_dist': 'Normal',
                                               'integration_window': 1./6,
                                               'n_variables': np.prod(env.observation_space.shape),
                                               'sigmoid_loc': True}

    model_args['obs_likelihood_args'] = {'type': 'fully_connected',
                                         'n_layers': 2,
                                         'n_input': n_state_variables + hidden_state_size,
                                         'n_units': 200,
                                         'connectivity': 'sequential',
                                         'batch_norm': False,
                                         'non_linearity': 'elu',
                                         'dropout': None}

    # reward
    model_args['reward_variable_args'] = {'type': 'fully_connected',
                                          'likelihood_dist': 'Normal',
                                          'integration_window': 0.1, # TODO: set this in a principled way
                                          'n_variables': 1,
                                          'sigmoid_loc': True}

    model_args['reward_likelihood_args'] = {'type': 'fully_connected',
                                            'n_layers': 1,
                                            'n_input': n_state_variables + hidden_state_size,
                                            'n_units': 200,
                                            'connectivity': 'sequential',
                                            'batch_norm': False,
                                            'non_linearity': 'elu',
                                            'dropout': None}

    # done
    model_args['done_variable_args'] = {'type': 'fully_connected',
                                        'likelihood_dist': 'Bernoulli',
                                        'n_variables': 1}

    model_args['done_likelihood_args'] = {'type': 'fully_connected',
                                          'n_layers': 1,
                                          'n_input': n_state_variables + hidden_state_size,
                                          'n_units': 100,
                                          'connectivity': 'sequential',
                                          'batch_norm': False,
                                          'non_linearity': 'elu',
                                          'dropout': None}

    model_args['misc_args'] = {'optimality_scale': 1e3,
                               'n_inf_iter': 1,
                               'kl_min': dict(state=0., action=0.)}

    return model_args
