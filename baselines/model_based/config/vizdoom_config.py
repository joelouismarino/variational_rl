import gym.spaces as spaces


def get_vizdoom_config(env):
    """
    Get the model configuration arguments for VizDoom environments.
    """
    model_args = {}

    action_space = env.action_space
    if type(action_space) == spaces.Discrete:
        # discrete control
        # TODO: used reparameterized categorical?
        action_prior_dist = 'Categorical'
        action_approx_post_dist = 'Categorical'
        n_action_variables = env.action_space.n
        action_inf_n_input = 2 * n_action_variables
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
                                      'n_layers': 2,
                                      'n_input': n_state_variables + n_action_variables,
                                      'n_units': 200,
                                      'connectivity': 'sequential',
                                      'dropout': None}

    model_args['state_inference_args'] = {'type': 'fully_connected',
                                          'n_layers': 2,
                                          'n_input': 4 * n_state_variables,
                                          'n_units': 500,
                                          'connectivity': 'sequential',
                                          'batch_norm': False,
                                          'non_linearity': 'elu',
                                          'dropout': None}

    # model_args['state_inference_args'] = {'type': 'vizdoom_skip_encoder',
    #                                       'non_linearity': 'elu'}

    # action
    model_args['action_variable_args'] = {'type': 'fully_connected',
                                          'prior_dist': action_prior_dist,
                                          'approx_post_dist': action_approx_post_dist,
                                          'n_variables': n_action_variables}

    model_args['action_prior_args'] = {'type': 'fully_connected',
                                       'n_layers': 2,
                                       'n_input': n_state_variables + n_action_variables,
                                       'n_units': 200,
                                       'connectivity': 'sequential',
                                       'batch_norm': False,
                                       'non_linearity': 'relu',
                                       'dropout': None}

    model_args['action_inference_args'] = {'type': 'fully_connected',
                                           'n_layers': 2,
                                           'n_input': action_inf_n_input,
                                           'n_units': 200,
                                           'connectivity': 'sequential',
                                           'batch_norm': False,
                                           'non_linearity': 'relu',
                                           'dropout': None}

    # observation
    model_args['observation_variable_args'] = {'type': 'transposed_conv',
                                               'likelihood_dist': 'Normal',
                                               'n_variables': env.observation_space.shape[1],
                                               'filter_size': 6,
                                               'padding': 0,
                                               'stride': 2}

    model_args['obs_likelihood_args'] = {'type': 'vizdoom_skip_decoder',
                                         'n_input': n_state_variables + n_action_variables,
                                         'non_linearity': 'elu'}

    # reward
    model_args['reward_variable_args'] = {'type': 'fully_connected',
                                          'likelihood_dist': 'Normal',
                                          'n_variables': 1}

    model_args['reward_likelihood_args'] = {'type': 'fully_connected',
                                            'n_layers': 2,
                                            'n_input': n_state_variables + n_action_variables,
                                            'n_units': 200,
                                            'connectivity': 'sequential',
                                            'batch_norm': False,
                                            'non_linearity': 'relu',
                                            'dropout': None}

    model_args['misc_args'] = {'optimality_scale': 1e3}

    return model_args
