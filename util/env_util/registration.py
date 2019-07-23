from gym.envs.registration import register

def register_env(env_name, cfg=None):
    '''Register additional environments for OpenAI gym.'''

    if env_name.lower() == 'vizdoom-v0':
        assert cfg is not None, 'Environment config name must be defined for vizdoom.'
        register(id='VizDoom-v0',
                 entry_point='envs.vizdoom.vizdoom_env:VizDoomEnv',
                 kwargs={'cfg_name': str(cfg)})
        return 'vizdoom'
