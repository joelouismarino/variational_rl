import numpy as np
import matplotlib.pyplot as plt


class Plot_Matplotlib:

    def __init__(self, path):
        self.path = path
        self.results = pu.load_results(path)
        self.plot_avg_rewards()

    def plot_avg_rewards(self):
        from baselines.common import plot_util as pu
        r = self.results[0]
        plt.figure(figsize=(6,4))
        x = np.cumsum(r.monitor.l)
        plt.plot(x, r.monitor.r) # raw data
        plt.plot(x, pu.smooth(r.monitor.r, radius=15)) # smoothed
        plt.xlabel('Timesteps')
        plt.ylabel('Average Reward')
        ax = plt.axes()
        plt.xlim(0,)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(f'{self.path}/average_reward.png', dpi = 300)

class Plot_visdom:

    def __init__(self, vis, log_path):
        self.vis = vis
        self.log_path = log_path
        self.avgrewardplot = None

    def plot(self):
        self.plot_avg_reward()

    def plot_avg_reward(self):
        from baselines.common import plot_util as pu
        results = pu.load_results(self.log_path)
        r = results[0]
        x = np.cumsum(r.monitor.l)
        y_raw = r.monitor.r
        y_smoothed = pu.smooth(r.monitor.r, radius=20)
        plt.plot(x, y_raw) # raw data
        plt.plot(x, y_smoothed) # smoothed
        if self.avgrewardplot is not None:
            self.vis.line(X = x, Y = y_raw, update = 'replace', name = 'All data', win = self.avgrewardplot)
            self.vis.line(X = x, Y = y_smoothed, update = 'replace', name = 'Smoothed', win = self.avgrewardplot)
        else:
            if len(x) > 0:
                self.avgrewardplot = self.vis.line(X = x, Y = y_raw, name = 'All data',
                                                   opts=dict(
                                                       xlabel='Timestep',
                                                       ylabel='Average Reward',
                                                       width=450,
                                                       height=320,
                                                       title=f"Rewards"
                                                   )
                                                   )