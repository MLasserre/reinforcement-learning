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

epsilons = [0., 0.1, 0.01]
results = {}

oa = '\\% Optimal action'
ar = 'Average reward'

for epsilon in epsilons:
    print("Epsilon: ", epsilon)
    runs = {oa:[], ar:[]}
    for i in range(n_run):
        #q = constant_initialization(k) # True values
        #B = bandit(k, q, lambda q: GRW_evolution(q, 0.01))
        #learner = epsilon_greedy_solver(B, epsilon, n_time, 
        #                         lambda N: 10, debug=True)
        q = bd.normal_initialization(k) # True values
        B = bd.bandit(k, q)
        learner = bl.epsilon_greedy_learner(B, epsilon, n_time, debug=True)
        learner.learn()
        infos = learner.getInfo()

        runs[oa].append(infos[0])
        runs[ar].append(infos[1])

    runs[oa] = np.array(runs[oa]).mean(axis=0)*100
    runs[ar] = np.array(runs[ar]).mean(axis=0)

    results[epsilon] = runs

fig, ax = plt.subplots()
for epsilon in results:
    ax.plot(list(range(n_time)), results[epsilon][ar],
            label='$\\epsilon = ' + str(epsilon) + '$', linewidth=1)
ax.set_ylabel(ar)
ax.set_xlim([0,n_time])
ax.set_xlabel('Steps')
ax.legend()
if save:
    fig.savefig("average_reward.pdf", transparent=True)
else:
    plt.show()

fig, ax= plt.subplots()
for epsilon in results:
    ax.plot(list(range(n_time)), results[epsilon][oa],
            label='$\\epsilon = ' + str(epsilon) + '$', linewidth=1)
ax.set_ylabel(oa)
ax.set_xlabel('Steps')
ax.set_xlim([0,n_time])
ax.set_ylim([0,100])
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend()
if save:
    fig.savefig("optimal_action.pdf", transparent=True)
else:
    plt.show()
