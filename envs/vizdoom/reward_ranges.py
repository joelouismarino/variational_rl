reward_ranges = {
    'basic': {'live': -1, 'shoot':-5, 'kill': 106}
}

# TODO: very hacky, tailored for basic
def get_reward_range(env_name, repeat):
    rewards = reward_ranges[env_name]
    min_reward = repeat * (rewards['live'] + rewards['shoot'])
    max_reward = rewards['live'] + rewards['shoot'] + rewards['kill']
    return [min_reward, max_reward]
