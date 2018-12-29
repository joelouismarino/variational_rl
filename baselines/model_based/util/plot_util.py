from visdom import Visdom
import numpy as np
vis = Visdom()


class PlotVisdom:

    def __init__(self):
        plots = ['free_energy',
                   'kl',
                   'cll',
                   'recon',
                   'obs']
        self.init_windows(plots)

    def init_windows(self, plots):
        self.window_id = {}
        for p in plots:
            self.window_id[p] = None


    def plot_visdom(self, log):
        self.plot_kl(log)
        self.plot_cll(log)
        self.plot_free_energy(log)


    def plot_kl(self, log):
        if self.window_id['kl'] is not None:
            vis.line(X=log['Step'], Y=log['KL'], update='replace', name='KL', win=self.window_id['kl'])
            # vis.line(X=x, Y=y_smoothed, update='replace', name='Smoothed', win=self.FreeEnergy)
            vis.line(X=log['Step'], Y=log['State KL'], update='replace', name='State KL', win=self.window_id['kl'])
            vis.line(X=log['Step'], Y=log['Action KL'], update='replace', name='Action KL', win=self.window_id['kl'])
        else:
            self.window_id['kl'] = vis.line(X=log['Step'], Y=log['KL'], name='KL',
                                            opts=dict(
                                           xlabel='Timestep',
                                           ylabel='KL',
                                           width=450,
                                           height=320,
                                            )
                                            )
            vis.line(X=log['Step'], Y=log['State KL'], update = 'append', name='State KL', win=self.window_id['kl'])
            vis.line(X=log['Step'], Y=log['Action KL'], update = 'append', name='Action KL', win=self.window_id['kl'])

    def plot_free_energy(self, log):
        if self.window_id['free_energy'] is not None:
            vis.line(X=log['Step'], Y=log['Free Energy'], update='replace', name='Free Energy', win=self.window_id['free_energy'])
            # vis.line(X=x, Y=y_smoothed, update='replace', name='Smoothed', win=self.FreeEnergy)
        else:
            self.window_id['free_energy'] = vis.line(X=log['Step'], Y=log['Free Energy'], name='Free Energy',
                                                     opts=dict(
                                                               xlabel='Timestep',
                                                               ylabel='Free Energy',
                                                               width=450,
                                                               height=320,
                                                     )
                                                     )

    def plot_cll(self, log):
        if self.window_id['cll'] is not None:
            vis.line(X=log['Step'], Y=log['CLL'], update='replace', name='CLL', win=self.window_id['cll'])
            # vis.line(X=x, Y=y_smoothed, update='replace', name='Smoothed', win=self.FreeEnergy)
            vis.line(X=log['Step'], Y=log['Reward CLL'], update = 'replace', name='Reward', win=self.window_id['cll'])
            vis.line(X=log['Step'], Y=log['Optimality CLL'], update = 'replace', name='Optimality', win=self.window_id['cll'])
            vis.line(X=log['Step'], Y=log['Obs CLL'], update = 'replace', name='Observation', win=self.window_id['cll'])

        else:
            self.window_id['cll'] = vis.line(X=log['Step'], Y=log['CLL'], name='CLL',
                                             opts=dict(
                                           xlabel='Timestep',
                                           ylabel='Conditional Log Likelihood',
                                           width=450,
                                           height=320,
                                             )
                                             )
            vis.line(X=log['Step'], Y=log['Reward CLL'], update = 'append', name='Reward', win=self.window_id['cll'])
            vis.line(X=log['Step'], Y=log['Optimality CLL'], update = 'append', name='Optimality', win=self.window_id['cll'])
            vis.line(X=log['Step'], Y=log['Obs CLL'], update = 'append', name='Observation', win=self.window_id['cll'])

    def visualize_obs_visdom(self, raw_img):
        title = 'Observation'
        size = 200
        self.window_id['obs'] = self.img_visdom(self.window_id['obs'], raw_img, title, size)

    def visualize_recon_visdom(self, raw_img):
        title = 'Reconstruction'
        size = 200
        self.window_id['recon'] = self.img_visdom(self.window_id['recon'], raw_img, title, size)


    def img_visdom(self, window, raw_img, title, size):
        img = self._preprocess_img(raw_img)
        if window is not None:
            vis.image(img, win = window,
                      opts=dict(
                          width = size,
                          height = size,
                          title = title
                      )
                      )
        else:
            window = vis.image(img,
                                 opts = dict(
                                     width = size,
                                     height = size,
                                     title = title
                                 )
                                 )
        return window

    def _preprocess_img(self, img):
        if type(img) != np.ndarray:
            # convert from torch
            img = img.detach().cpu().numpy()
        if len(img.shape) == 4:
            # remove batch dimension
            img = img[0]
        return img


