

def postprocess_misc_args(misc_args):
    """
    Postprocesses the misc_args dict to include any new keys.
    """

    new_keys = ['optimize_targets',
                'model_value_targets',
                'direct_targets',
                'off_policy_targets',
                'target_inf_value_targets',
                'inf_target_kl',
                'critic_grad_penalty',
                'pessimism']

    for key in new_keys:
        if key not in misc_args:
            if key == 'critic_grad_penalty':
                misc_args[key] = 0.
            elif key == 'pessimism':
                misc_args[key] = 1.
            else:
                misc_args[key] = False

    if 'use_target_inference_optimizer' in misc_args:
        if misc_args['use_target_inference_optimizer']:
            misc_args['target_inf_value_targets'] = True

    return misc_args
