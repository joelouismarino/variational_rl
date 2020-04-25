
def calculate_n_inputs(inputs, config_dict):
    """
    Calculate the number of inputs for a particular model.
    """
    input_size = 0
    for input_name in inputs:
        if input_name == 'action':
            input_size += config_dict['prior_args']['n_variables']
        elif input_name == 'state':
            input_size += config_dict['misc_args']['state_size']
        elif input_name == 'reward':
            input_size += 1
        elif input_name in ['params', 'grads']:
            input_size += 2 * config_dict['prior_args']['n_variables']
    return input_size

def get_n_input(config_dict):
    """
    Calculate the number of inputs for each model using the inputs list.
    """
    model_dicts = []

    # Q-network
    model_dicts.append(config_dict['q_value_estimator_args']['network_args'])

    # state value network
    if config_dict['state_value_estimator_args'] is not None:
        model_dicts.append(config_dict['state_value_estimator_args']['network_args'])

    # prior network
    if config_dict['prior_model_args'] is not None:
        model_dicts.append(config_dict['prior_model_args'])

    # inference optimizer network
    if config_dict['inference_optimizer_args']['opt_type'] in ['direct', 'iterative']:
        model_dicts.append(config_dict['inference_optimizer_args']['network_args'])

    # direct inference optimizer network
    if config_dict['direct_inference_optimizer_args'] is not None:
        model_dicts.append(config_dict['direct_inference_optimizer_args']['network_args'])

    # model network(s)
    if config_dict['q_value_estimator_args']['estimator_type'] == 'model_based':
        model_dicts.append(config_dict['q_value_estimator_args']['model_args']['state_likelihood_args'])
        if config_dict['q_value_estimator_args']['learn_reward']:
            model_dicts.append(config_dict['q_value_estimator_args']['model_args']['reward_likelihood_args'])

    for model_dict in model_dicts:
        inputs = model_dict['inputs']
        input_size = calculate_n_inputs(inputs, config_dict)
        model_dict['n_input'] = input_size

    return config_dict
