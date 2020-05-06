import numpy as np
from .get_n_input import get_n_input
from .get_euler_args import get_euler_args

def get_mujoco_config(env):
    """
    Get the model configuration arguments for MuJoCo environments.
    """
    agent_args = {}

    agent_args['misc_args'] = {'n_action_samples': 10,
                               'n_q_action_samples': 1,
                               'reward_discount': 0.99,
                               'retrace_lambda': 0.9,
                               'postprocess_action': False,
                               'epsilons': dict(pi=None, loc=5e-4, scale=1e-5)}
                               # RERPI epsilons: pi=0.1, loc=5e-4, scale=1e-5
                               # use pi=None for SAC heuristic

    state_size = int(np.prod(env.observation_space.shape))
    agent_args['misc_args']['state_size'] = state_size
    n_action_variables = env.action_space.shape[0]

    # distribution types: 'Uniform', 'Normal', 'TanhNormal', 'Boltzmann', 'NormalUniform'
    action_prior_dist = 'Uniform'
    action_approx_post_dist = 'TanhNormal'

    ## PRIOR
    constant_prior = True
    agent_args['prior_args'] = {'dist_type': action_prior_dist,
                                'n_variables': n_action_variables,
                                'constant': constant_prior}

    if action_prior_dist in ['ARNormal', 'TanhARNormal']:
        agent_args['prior_args']['transform_config'] = {'n_transforms': 1,
                                                        'type': 'ar_fully_connected',
                                                        'n_layers': 2,
                                                        'n_input': n_action_variables,
                                                        'n_units': n_action_variables,
                                                        'non_linearity': 'elu'}

    if action_prior_dist == 'Uniform' or constant_prior:
        agent_args['prior_model_args'] = None
        agent_args['prior_args']['constant'] = True
    else:
        agent_args['prior_model_args'] = {'type': 'fully_connected',
                                          'n_layers': 3,
                                          'inputs': ['state'],
                                          'n_units': 256,
                                          'connectivity': 'sequential',
                                          'batch_norm': False,
                                          'non_linearity': ['tanh', 'elu', 'elu'],
                                          'layer_norm': [True, False, False],
                                          'dropout': None,
                                          'separate_networks': False}

    ## APPROXIMATE POSTERIOR
    agent_args['approx_post_args'] = {'dist_type': action_approx_post_dist,
                                      'n_variables': n_action_variables,
                                      'constant_scale': False}

    if action_approx_post_dist in ['ARNormal', 'TanhARNormal']:
        agent_args['approx_post_args']['transform_config'] = {'n_transforms': 1,
                                                              'type': 'ar_fully_connected',
                                                              'n_layers': 2,
                                                              'n_input': n_action_variables,
                                                              'n_units': n_action_variables,
                                                              'non_linearity': 'elu'}

    ## INFERENCE OPTIMIZER
    # optimizer type can be 'direct', 'iterative', 'gradient', 'non_parametric', 'cem'
    optimizer_type = 'direct'
    optimizer_type = 'non_parametric' if action_approx_post_dist == 'Boltzmann' else optimizer_type
    use_direct_inference_optimizer = True
    agent_args['misc_args']['use_target_inference_optimizer'] = False

    inf_opt_args = {'opt_type': optimizer_type}
    if optimizer_type == 'direct':
        agent_args['approx_post_args']['update'] = 'direct'
        inf_opt_args['network_args'] = {'type': 'fully_connected',
                                                'n_layers': 2,
                                                'inputs': ['state'],
                                                'n_units': 256,
                                                'connectivity': 'sequential',
                                                'batch_norm': False,
                                                'non_linearity': 'relu',
                                                'dropout': None,
                                                'separate_networks': False}
    elif optimizer_type == 'iterative':
        inf_opt_args['n_inf_iters'] = 5
        agent_args['approx_post_args']['update'] = 'iterative'
        inf_opt_args['network_args'] = {'type': 'fully_connected',
                                                'n_layers': 2,
                                                'inputs': ['state', 'params', 'grads'],
                                                'n_units': 256,
                                                'connectivity': 'sequential',
                                                'batch_norm': False,
                                                'non_linearity': 'relu',
                                                'dropout': None,
                                                'separate_networks': False}
    elif optimizer_type == 'gradient':
        inf_opt_args['n_inf_iters'] = 10
        inf_opt_args['lr'] = 3e-4
    elif optimizer_type == 'non_parametric':
        assert action_approx_post_dist == 'Boltzmann'
    elif optimizer_type == 'cem':
        assert action_approx_post_dist == 'Normal'
        inf_opt_args['n_top_samples'] = 10
        inf_opt_args['n_inf_iters'] = 3

    agent_args['inference_optimizer_args'] = inf_opt_args

    if use_direct_inference_optimizer:
        # use a direct inference model, for planning and/or estimating value targets
        agent_args['misc_args']['direct_targets'] = True
        agent_args['direct_approx_post_args'] = {'dist_type': action_approx_post_dist,
                                                 'n_variables': n_action_variables,
                                                 'update': 'direct'}
        inf_opt_args = {'opt_type': 'direct'}
        inf_opt_args['network_args'] = {'type': 'fully_connected',
                                                'n_layers': 2,
                                                'inputs': ['state'],
                                                'n_units': 256,
                                                'connectivity': 'sequential',
                                                'batch_norm': False,
                                                'non_linearity': 'relu',
                                                'dropout': None,
                                                'separate_networks': False}
        agent_args['direct_inference_optimizer_args'] = inf_opt_args
    else:
        agent_args['direct_approx_post_args'] = None
        agent_args['direct_inference_optimizer_args'] = None
        agent_args['misc_args']['direct_targets'] = False

    ## Q-VALUE ESTIMATOR
    # estimator type can be 'direct' or 'model_based'
    estimator_type = 'direct'

    # whether to use a separate state-value network
    use_state_value_network = True

    if use_state_value_network:
        state_value_args = {}
        state_value_args['network_args'] = {'type': 'fully_connected',
                                            'n_layers': 2,
                                            'inputs': ['state'],
                                            'n_units': 512,
                                            'connectivity': 'sequential',
                                            'non_linearity': 'relu',
                                            'layer_norm': False,
                                            'dropout': None}
        agent_args['state_value_estimator_args'] = state_value_args
    else:
        agent_args['state_value_estimator_args'] = None

    # whether to use target networks for policy optimization
    agent_args['misc_args']['optimize_targets'] = True

    # whether to use the model for value network targets
    if estimator_type == 'model_based':
        agent_args['misc_args']['model_value_targets'] = False
    else:
        agent_args['misc_args']['model_value_targets'] = False

    estimator_args = {'estimator_type': estimator_type}
    estimator_args['network_args'] = {'type': 'fully_connected',
                                      'n_layers': 3,
                                      'inputs': ['state', 'action'],
                                      'n_units': 512,
                                      'connectivity': 'highway',
                                      'non_linearity': 'elu',
                                      'layer_norm': True,
                                      'dropout': None}
    if estimator_type == 'model_based':
        learn_reward = True
        value_estimate = 'retrace'
        use_euler = True
        stochastic_state = False
        stochastic_reward = False
        model_args = {}
        model_args['state_likelihood_args'] = {'type': 'fully_connected',
                                                       'n_layers': 2,
                                                       'inputs': ['state', 'action'],
                                                       'n_units': 256,
                                                       'connectivity': 'sequential',
                                                       'batch_norm': False,
                                                       'layer_norm': True,
                                                       'non_linearity': 'leaky_relu'}
        model_args['state_variable_args'] = {'type': 'fully_connected',
                                                     'likelihood_dist': 'Normal',
                                                     'n_variables': state_size,
                                                     'stochastic': stochastic_state,
                                                     'constant_scale': True,
                                                     'residual_loc': False,
                                                     'euler_loc': use_euler}
        if use_euler:
            model_args['state_variable_args']['euler_args'] = get_euler_args(env)
        if learn_reward:
            model_args['reward_likelihood_args'] = {'type': 'fully_connected',
                                                            'n_layers': 2,
                                                            'inputs': ['state', 'action'],
                                                            'n_units': 256,
                                                            'connectivity': 'sequential',
                                                            'batch_norm': False,
                                                            'layer_norm': True,
                                                            'non_linearity': 'leaky_relu'}
            model_args['reward_variable_args'] = {'type': 'fully_connected',
                                                          'likelihood_dist': 'Normal',
                                                          'n_variables': 1,
                                                          'stochastic': stochastic_reward,
                                                          'constant_scale': True,
                                                          'residual_loc': False}
        estimator_args['model_args'] = model_args
        estimator_args['learn_reward'] = learn_reward
        estimator_args['value_estimate'] = value_estimate
        estimator_args['horizon'] = 2

    agent_args['q_value_estimator_args'] = estimator_args

    # calculate the input sizes for all models
    agent_args = get_n_input(agent_args)
    return agent_args
