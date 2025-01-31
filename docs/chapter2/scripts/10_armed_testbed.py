import numpy as np
import reinforcement_learning.chapter2.bandit as bd

import os

import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')

output_dir = 'docs/chapter2/output/figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

k = 10
n_run = 2000
n_step = 1000

epsilons = {0.: 'green', 0.1: 'blue', 0.01: 'red'}
results = {}

oa = '\\% Optimal action'
ar = 'Average reward'

for epsilon in epsilons:
    print("Epsilon: ", epsilon)
    runs = {oa:[], ar:[]}
    for i in range(n_run):
        B = bd.Bandit(k)
        learner = bd.EpsilonGreedyLearner(B, epsilon, debug=True)
        learner.learn(n_step)
        infos = learner.getInfo()

        runs[oa].append(infos[0])
        runs[ar].append(infos[1])

    runs[oa] = np.array(runs[oa]).mean(axis=0)*100
    runs[ar] = np.array(runs[ar]).mean(axis=0)

    results[epsilon] = runs


# Plotting the figure for the average reward
fig, ax = plt.subplots()
for epsilon in results:
    ax.plot(list(range(n_step)),
            results[epsilon][ar],
            label='$\\epsilon = ' + str(epsilon) + '$',
            color=epsilons[epsilon],
            linewidth=1)

ax.set_ylabel(ar)
ax.set_xlim([0,n_step])
ax.set_xlabel('Steps')
ax.legend()
fig_path = os.path.join(output_dir, "average_reward.pdf")
fig.savefig(fig_path, transparent=True)

# Plotting the figure for the optimal action
fig, ax= plt.subplots()
for epsilon in results:
    ax.plot(list(range(n_step)),
            results[epsilon][oa],
            label='$\\epsilon = ' + str(epsilon) + '$',
            color=epsilons[epsilon],
            linewidth=1)
ax.set_ylabel(oa)
ax.set_xlabel('Steps')
ax.set_xlim([0,n_step])
ax.set_ylim([0,100])
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend()
fig_path = os.path.join(output_dir, "optimal_action.pdf")
fig.savefig(fig_path, transparent=True)
