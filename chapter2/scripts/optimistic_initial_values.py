import numpy as np
import bandit as bd
import bandit_learner as bl

import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')

save = True

k = 10
n_run = 2000
n_time = 1000

methods = {}
methods['realistic'] = {'Q0': 0., 'eps': 0.1, 'color': 'gray'}
methods['optimistic'] = {'Q0': 5., 'eps': 0., 'color': 'blue'}

results = {}

oa = '\\% Optimal action'
ar = 'Average reward'


for method in methods:
    print("Method: ", method)
    runs = {oa:[], ar:[]}
    for i in range(n_run):
        B = bd.Bandit(k)
        learner = bl.EpsilonGreedyLearner(B, methods[method]['eps'], debug=True)
        learner.setInitalValues(np.full(k, methods[method]['Q0']))
        learner.setUpdateMethod(lambda N: 0.1)
        learner.learn(n_time)
        infos = learner.getInfo()

        runs[oa].append(infos[0])
        runs[ar].append(infos[1])

    runs[oa] = np.array(runs[oa]).mean(axis=0)*100
    runs[ar] = np.array(runs[ar]).mean(axis=0)

    results[method] = runs


# Plotting the figure for the average reward
fig, ax = plt.subplots()
for method in results:
    ax.plot(list(range(n_time)),
            results[method][ar],
            label=f'${method}$',
            color=methods[method]['color'],
            linewidth=1)

ax.set_ylabel(ar)
ax.set_xlim([0,n_time])
ax.set_xlabel('Steps')
ax.legend()
if save:
    fig.savefig("optimistic_average_reward.pdf", transparent=True)
else:
    plt.show()

# Plotting the figure for the optimal action
fig, ax= plt.subplots()
for method in results:
    ax.plot(list(range(n_time)),
            results[method][oa],
            label=f'${method}$',
            color=methods[method]['color'],
            linewidth=1)
ax.set_ylabel(oa)
ax.set_xlabel('Steps')
ax.set_xlim([0,n_time])
ax.set_ylim([0,100])
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend()
if save:
    fig.savefig("optimistic_optimal_action.pdf", transparent=True)
else:
    plt.show()
