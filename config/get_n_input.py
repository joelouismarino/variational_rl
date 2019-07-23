
def get_n_input(config_dict, discrete_actions):
    """
    Calculate the number of inputs for each model using the inputs list.
    """
    model_names = ['state_prior', 'state_inference', 'action_prior',
                   'action_inference', 'value_model']

    if config_dict['agent_type'] == 'generative':
        model_names += ['obs_likelihood', 'reward_likelihood',
                        'done_likelihood']

    for model_name in model_names:
        input_size = 0
        inputs = config_dict[model_name + '_args']['inputs']

        for input_name in inputs:
            if input_name == 'state':
                input_size += config_dict['state_variable_args']['n_variables']
            elif input_name == 'action':
                input_size += config_dict['action_variable_args']['n_variables']
            elif input_name == 'observation':
                input_size += config_dict['observation_variable_args']['n_variables']
            elif input_name == 'reward':
                input_size += 1
            elif input_name in ['params', 'grads']:
                if 'state' in model_name:
                    input_size += 2 * config_dict['state_variable_args']['n_variables']
                else:
                    if discrete_actions:
                        input_size += config_dict['state_variable_args']['n_variables']
                    else:
                        input_size += 2 * config_dict['state_variable_args']['n_variables']

        config_dict[model_name + '_args']['n_input'] = input_size

    return config_dict
