import gym.spaces as spaces


def get_vizdoom_config(env):
    """
    Get the model configuration arguments for VizDoom environments.
    """
    agent_args = {}

    agent_args['agent_type'] = 'discriminative'

    agent_args['misc_args'] = {'optimality_scale': 1,
                               'n_inf_iter': dict(state=1, action=1),
                               'kl_min': dict(state=0., action=0.75),
                               'gae_lambda': 0.95}

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

    if agent_args['agent_type'] == 'discriminative':
        # state
        n_state_variables = 100
        agent_args['state_variable_args'] = {'type': 'fully_connected',
                                             'prior_dist': 'Normal',
                                             'approx_post_dist': None,
                                             'n_variables': n_state_variables}

        agent_args['state_prior_args'] = {'type': 'vizdoom_skip_encoder',
                                          'non_linearity': 'relu'}

        agent_args['state_inference_args'] = None

        # action
        agent_args['action_variable_args'] = {'type': 'fully_connected',
                                              'prior_dist': action_prior_dist,
                                              'approx_post_dist': action_approx_post_dist,
                                              'n_variables': n_action_variables,
                                              'constant_prior': True,
                                              'inference_type': 'direct'}

        agent_args['action_prior_args'] = None

        agent_args['action_inference_args'] = {'type': 'fully_connected',
                                               'n_layers': 1,
                                               'n_input': n_state_variables,
                                               'n_units': 64,
                                               'connectivity': 'sequential',
                                               'batch_norm': False,
                                               'non_linearity': 'tanh',
                                               'dropout': None}

        # value
        agent_args['value_model_args'] = {'type': 'fully_connected',
                                          'n_layers': 1,
                                          'n_input': n_state_variables,
                                          'n_units': 64,
                                          'connectivity': 'sequential',
                                          'non_linearity': 'tanh',
                                          'dropout': None}

    elif agent_args['agent_type'] == 'generative':

        # observation
        agent_args['observation_variable_args'] = {'type': 'transposed_conv',
                                                   'likelihood_dist': 'Normal',
                                                   'integration_window': 1./256,
                                                   'n_variables': env.observation_space.shape[1],
                                                   'filter_size': 6,
                                                   'padding': 0,
                                                   'stride': 2,
                                                   'sigmoid_loc': True}

        agent_args['obs_likelihood_args'] = {'type': 'vizdoom_skip_decoder',
                                             'n_input': n_state_variables + hidden_state_size,
                                             'non_linearity': 'elu'}

        # reward
        agent_args['reward_variable_args'] = {'type': 'fully_connected',
                                              'likelihood_dist': 'Normal',
                                              'integration_window': 0.1, # TODO: set this in a principled way
                                              'n_variables': 1,
                                              'sigmoid_loc': True}

        agent_args['reward_likelihood_args'] = {'type': 'fully_connected',
                                                'n_layers': 2,
                                                'n_input': n_state_variables + hidden_state_size,
                                                'n_units': 200,
                                                'connectivity': 'sequential',
                                                'batch_norm': False,
                                                'non_linearity': 'elu',
                                                'dropout': None}

        # done
        agent_args['done_variable_args'] = {'type': 'fully_connected',
                                            'likelihood_dist': 'Bernoulli',
                                            'n_variables': 1}

        agent_args['done_likelihood_args'] = {'type': 'fully_connected',
                                              'n_layers': 1,
                                              'n_input': n_state_variables + hidden_state_size,
                                              'n_units': 100,
                                              'connectivity': 'sequential',
                                              'batch_norm': False,
                                              'non_linearity': 'elu',
                                              'dropout': None}

    return agent_args
