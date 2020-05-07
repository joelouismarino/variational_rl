from gym.envs.registration import register

def register_env(env_name):
    '''Register additional environments for OpenAI gym.'''

    if env_name == 'AntTruncatedObs-v2':
        register(id='AntTruncatedObs-v2',
                 entry_point='util.env_util.mujoco.ant:AntTruncatedObsEnv')

    elif env_name == 'HumanoidTruncatedObs-v2':
        register(id='HumanoidTruncatedObs-v2',
                 entry_point='util.env_util.mujoco.humanoid:HumanoidTruncatedObsEnv')

    elif env_name == 'Drone-v0':
        register(id='Drone-v0',
                 entry_point='util.env_util.drone.drone:DroneEnv')
