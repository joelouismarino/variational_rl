

def postprocess_misc_args(misc_args):
    """
    Postprocesses the misc_args dict to include any new keys.
    """

    new_keys = ['use_target_inference_optimizer',
                'optimize_targets',
                'model_value_targets',
                'direct_targets',
                'off_policy_targets',
                'use_target_inference_optimizer']

    for key in new_keys:
        if key not in misc_args:
            misc_args[key] = False

    return misc_args
