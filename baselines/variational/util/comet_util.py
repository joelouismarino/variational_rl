import pandas as pd
import matplotlib.pyplot as plt


def plot_ts(actions, experiment, step, name):
    nb_actions = min(9, actions.shape[1])
    k = 1
    j = str(nb_actions)
    for i in range(nb_actions):
        plt.subplot(int(j + '1' + str(k)))
        plt.plot(actions[i])
        k += 1

    experiment.log_figure(figure=plt, figure_name=name + '_ts_'+str(step))
    plt.close()


def flatten(dictionary):
    return pd.io.json.json_normalize(dictionary, sep='_').to_dict()


def plot_episode(episode, experiment, step):
    episode = flatten(episode)
    for n, m in episode.items():
        if len(m[0]) > 0:
            m = pd.DataFrame(m[0].cpu().numpy())
            plot_ts(m, experiment, step, n)