reward_ranges = {
    'basic': {'live': -1, 'shoot':-5, 'kill': 106},
    'take_cover': {'live': 0.1, 'shoot': 0, 'kill': 0}
}

# TODO: very hacky, tailored for basic
def get_reward_range(env_name, repeat):
    rewards = reward_ranges[env_name]
    if rewards['live'] < 0:
        min_reward = repeat * (rewards['live'] + rewards['shoot'])
        max_reward = rewards['live'] + rewards['shoot'] + rewards['kill']
    else:
        min_reward = 0
        max_reward = repeat * rewards['live']
    return [min_reward, max_reward]
