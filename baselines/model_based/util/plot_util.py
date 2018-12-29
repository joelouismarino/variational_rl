from visdom import Visdom
vis = Visdom()


class Plot_visdom:

    def __init__(self):
        self.FreeEnergy = None
        self.KL = None
        self.CLL = None

    def plot_visdom(self, log):
        self.plot_KL(log)
        self.plot_CLL(log)
        self.plot_FreeEnergy(log)


    def plot_KL(self, log):
        if self.KL is not None:
            vis.line(X=log['Step'], Y=log['KL'], update='replace', name='KL', win=self.KL)
            # vis.line(X=x, Y=y_smoothed, update='replace', name='Smoothed', win=self.FreeEnergy)
            vis.line(X=log['Step'], Y=log['State KL'], update='replace', name='State KL', win=self.KL)
            vis.line(X=log['Step'], Y=log['Action KL'], update='replace', name='Action KL', win=self.KL)
        else:
            self.KL = vis.line(X=log['Step'], Y=log['KL'], name='KL',
                                       opts=dict(
                                           xlabel='Timestep',
                                           ylabel='KL',
                                           width=450,
                                           height=320,
                                       )
                                       )
            vis.line(X=log['Step'], Y=log['State KL'], update = 'append', name='State KL', win=self.KL)
            vis.line(X=log['Step'], Y=log['Action KL'], update = 'append', name='Action KL', win=self.KL)

    def plot_FreeEnergy(self, log):
        if self.FreeEnergy is not None:
            vis.line(X=log['Step'], Y=log['Free Energy'], update='replace', name='Free Energy', win=self.FreeEnergy)
            # vis.line(X=x, Y=y_smoothed, update='replace', name='Smoothed', win=self.FreeEnergy)
        else:
            self.FreeEnergy = vis.line(X=log['Step'], Y=log['Free Energy'], name='Free Energy',
                                       opts=dict(
                                           xlabel='Timestep',
                                           ylabel='Free Energy',
                                           width=450,
                                           height=320,
                                       )
                                       )

    def plot_CLL(self, log):
        if self.CLL is not None:
            vis.line(X=log['Step'], Y=log['CLL'], update='replace', name='CLL', win=self.CLL)
            # vis.line(X=x, Y=y_smoothed, update='replace', name='Smoothed', win=self.FreeEnergy)
            vis.line(X=log['Step'], Y=log['Reward CLL'], update = 'replace', name='Reward', win=self.CLL)
            vis.line(X=log['Step'], Y=log['Optimality CLL'], update = 'replace',  name='Optimality', win=self.CLL)
            vis.line(X=log['Step'], Y=log['Obs CLL'], update = 'replace',  name='Observation', win=self.CLL)

        else:
            self.CLL = vis.line(X=log['Step'], Y=log['CLL'], name='CLL',
                                       opts=dict(
                                           xlabel='Timestep',
                                           ylabel='Conditional Log Likelihood',
                                           width=450,
                                           height=320,
                                           showlegend=True
                                       )
                                       )
            vis.line(X=log['Step'], Y=log['Reward CLL'], update = 'append', name='Reward', win=self.CLL)
            vis.line(X=log['Step'], Y=log['Optimality CLL'], update = 'append', name='Optimality', win=self.CLL)
            vis.line(X=log['Step'], Y=log['Obs CLL'], update = 'append', name='Observation', win=self.CLL)
