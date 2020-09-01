import numpy as np
import matplotlib.pyplot as plt

k = 10
n_run = 2000
n_time = 1000

epsilons = [0, 0.1, 0.01]
results = {}

for epsilon in epsilons:
    print("Epsilon: ", epsilon)
    l_good_action = []
    l_total_reward = []
    for i in range(n_run):
        print("\tIteration ", i+1)
        q = np.random.normal(size=k)
        max_action = np.argmax(q)
        Q = np.zeros(k)
        n_selected = np.zeros(k, dtype=int)
        good_action = []
        total_reward = []
        for t in range(n_time):
            # print("\tTime ", t)
            is_greedy = np.random.uniform() > epsilon
            if is_greedy:
                index = np.argmax(Q) # Would be better with random ?
            else:
                index = np.random.randint(0, k)
            reward = np.random.normal(loc=q[index])
            Q[index] = (n_selected[index]*Q[index] + reward) / (n_selected[index] + 1)
            n_selected[index] += 1
            good_action.append(int(index == max_action))
            total_reward.append(reward)
        l_good_action.append(good_action)
        l_total_reward.append(total_reward)
    
    l_good_action = np.array(l_good_action)
    l_good_action = l_good_action.mean(axis=0)
    
    l_total_reward = np.array(l_total_reward)
    l_total_reward = l_total_reward.mean(axis=0)

    results[epsilon] = [l_total_reward, l_good_action]
    

fig, ax = plt.subplots()
for epsilon in results:
    ax.plot(list(range(n_time)), results[epsilon][0], label=str(epsilon))
ax.legend()
fig.savefig("average_reward.pdf", transparent=True)

fig, ax= plt.subplots()
for epsilon in results:
    ax.plot(list(range(n_time)), results[epsilon][1], label=str(epsilon))
ax.legend()
fig.savefig("optimal_action.pdf", transparent=True)

