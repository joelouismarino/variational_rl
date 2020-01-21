import numpy as np
from .get_n_input import get_n_input


def get_mujoco_config(env):
    """
    Get the model configuration arguments for MuJoCo environments.
    """
    agent_args = {}

    agent_args['misc_args'] = {'n_action_samples': 50,
                               'n_q_action_samples': 1,
                               'reward_discount': 0.99,
                               'retrace_lambda': 0.75,
                               'postprocess_action': False,
                               'epsilons': dict(pi=None, loc=5e-4, scale=1e-5)}
                               # RERPI epsilons: pi=0.1, loc=5e-4, scale=1e-5
                               # use pi=None for SAC heuristic

    state_size = np.prod(env.observation_space.shape)
    agent_args['misc_args']['state_size'] = state_size
    n_action_variables = env.action_space.shape[0]

    # distribution types: 'Uniform', 'Normal', 'TanhNormal', 'Boltzmann', 'NormalUniform'
    action_prior_dist = 'Uniform'
    action_approx_post_dist = 'TanhNormal'

    ## PRIOR
    constant_prior = False
    agent_args['prior_args'] = {'dist_type': action_prior_dist,
                                'n_variables': n_action_variables,
                                'constant': constant_prior}

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
                                      'n_variables': n_action_variables}

    ## INFERENCE OPTIMIZER
    # optimizer type can be 'direct', 'iterative', 'gradient', 'non_parametric', 'cem'
    optimizer_type = 'direct'
    optimizer_type = 'non_parametric' if action_approx_post_dist == 'Boltzmann' else optimizer_type

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
        inf_opt_args['n_inf_iters'] = 2
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
        inf_opt_args['lr'] = 1e-3
    elif optimizer_type == 'non_parametric':
        assert action_approx_post_dist == 'Boltzmann'
    elif optimizer_type == 'cem':
        assert action_approx_post_dist == 'Normal'
        inf_opt_args['n_top_samples'] = 10
        inf_opt_args['n_inf_iters'] = 3

    agent_args['inference_optimizer_args'] = inf_opt_args

    ## Q-VALUE ESTIMATOR
    # estimator type can be 'direct' or 'model_based'
    estimator_type = 'model_based'

    estimator_args = {'estimator_type': estimator_type}
    estimator_args['network_args'] = {'type': 'fully_connected',
                                      'n_layers': 2,
                                      'inputs': ['state', 'action'],
                                      'n_units': 256,
                                      'connectivity': 'sequential',
                                      'non_linearity': 'relu',
                                      'layer_norm': False,
                                      'dropout': None}
    if estimator_type == 'model_based':
        learn_reward = True
        value_estimate = 'retrace'
        stochastic_state = False
        stochastic_reward = False
        model_args = {}
        model_args['state_likelihood_args'] = {'type': 'fully_connected',
                                                       'n_layers': 2,
                                                       'inputs': ['state', 'action'],
                                                       'n_units': 256,
                                                       'connectivity': 'sequential',
                                                       'batch_norm': False,
                                                       'non_linearity': 'relu'}
        model_args['state_variable_args'] = {'type': 'fully_connected',
                                                     'likelihood_dist': 'Normal',
                                                     'n_variables': state_size,
                                                     'stochastic': stochastic_state,
                                                     'constant_scale': False,
                                                     'residual_loc': True}
        if learn_reward:
            model_args['reward_likelihood_args'] = {'type': 'fully_connected',
                                                            'n_layers': 2,
                                                            'inputs': ['state', 'action'],
                                                            'n_units': 256,
                                                            'connectivity': 'sequential',
                                                            'batch_norm': False,
                                                            'non_linearity': 'relu'}
            model_args['reward_variable_args'] = {'type': 'fully_connected',
                                                          'likelihood_dist': 'Normal',
                                                          'n_variables': 1,
                                                          'stochastic': stochastic_reward,
                                                          'constant_scale': False,
                                                          'residual_loc': False}
        estimator_args['model_args'] = model_args
        estimator_args['learn_reward'] = learn_reward
        estimator_args['value_estimate'] = value_estimate
        estimator_args['horizon'] = 5

    agent_args['q_value_estimator_args'] = estimator_args

    # calculate the input sizes for all models
    agent_args = get_n_input(agent_args)
    return agent_args
