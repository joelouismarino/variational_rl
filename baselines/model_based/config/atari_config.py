def get_atari_config(env):
    """
    Get the model configuration arguments for Atari environments.
    """
    model_args = {}

    # state
    state_variable_type = 'fully_connected'
    n_state_variables

    model_args['state_variable_args'] = {}

    model_args['state_prior_args'] = {}

    model_args['state_inference_args'] = {}

    # action
    action_variable_type = 'fully_connected'
    n_action_variables = env.action_space.n

    model_args['action_variable_args'] = {}

    model_args['action_prior_args'] = {}

    model_args['action_inference_args'] = {}

    # observation
    obs_variable_type = 'convolutional'
    obs_shape = env.observation_space.shape

    model_args['observation_variable_args'] = {}

    model_args['obs_likelihood_args'] = {}

    # reward
    reward_variable_type = 'fully_connected'

    model_args['reward_variable_args'] = {}

    model_args['reward_likelihood_args'] = {}

    return model_args
