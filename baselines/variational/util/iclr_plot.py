import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob

disc_paths = '/home/tzhang/Dropbox/Caltech/VisionLab/RL/baselines/data/model-based/iclr_plot_data/' \
             'steps/minigrid_normal_discriminative'
disc_exps = glob.glob(disc_paths+"/*.p")
print(len(disc_exps))

gen_paths = '/home/tzhang/Dropbox/Caltech/VisionLab/RL/baselines/data/model-based/iclr_plot_data/' \
            'steps/minigrid_generative_noplanning'
gen_noplan_exp = glob.glob(gen_paths+"/*.p")
print(len(gen_noplan_exp))

nb_episodes_plot = 80

disc_steps = []
for e in disc_exps:
    print(e)
    eval_stats = pickle.load(open(e, 'rb'))
    steps = eval_stats['episode length'][:nb_episodes_plot]
    disc_steps.append(steps)
disc_steps = np.array(disc_steps)

gen_noplan_steps = []
for e in gen_noplan_exp:
    eval_stats = pickle.load(open(e, 'rb'))
    print(eval_stats.keys())
    steps = eval_stats['episode length'][:nb_episodes_plot]
    gen_noplan_steps.append(steps)
gen_noplan_steps = np.array(gen_noplan_steps)

gen_plan_steps = []
gen_plan_exps = pickle.load(open('/home/tzhang/Dropbox/Caltech/VisionLab/RL/baselines/'
                                 'data/model-based/iclr_plot_data/'
                                 'steps/minigrid_generative_planning/planning_results.p','rb')) # from joe
for e in gen_plan_exps.values():
    steps = e[:nb_episodes_plot]
    gen_plan_steps.append(steps)


labels = ['Discriminative', 'Generative (non-planning)', 'Generative (planning)']

data = [disc_steps, gen_noplan_steps, gen_plan_steps]


# plot

from matplotlib import rc
from matplotlib import rc
import matplotlib.pylab as plt

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



plt.figure(figsize=(6, 4))
x = np.arange(1, nb_episodes_plot+1)
# averages = np.average(data, axis=1)
averages = []
for i in data:
    averages.append(np.average(i,axis=0))
    print(len(i))

errors = []
for i in data:
    errors.append(np.std(i,axis = 0))

for exp_nb, (avg_i, err_i, label) in enumerate(zip(averages, errors, labels)):
    upper_conf, lower_conf = avg_i + err_i, avg_i - err_i
    plt.fill_between(x, lower_conf, upper_conf, alpha=0.2)
    plt.plot(x, avg_i, linewidth=1.5, label=label)
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Episode', fontsize = 15)
plt.xlim(1, nb_episodes_plot)
plt.ylim(0, )
plt.ylabel('Episode Length', fontsize = 15)
plt.legend(frameon=False, loc='best', fontsize='medium')
plt.savefig('/home/tzhang/Dropbox/Caltech/VisionLab/RL/baselines/data/model-based/iclr_plot_data/'
            'iclr_gridworld_8x8_comparisons.pdf',
            dpi=300, bbox_inches='tight')