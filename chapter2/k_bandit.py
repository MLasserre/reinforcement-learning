import numpy as np

import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')

def normal_initialization(k, mu=0., sigma=1.):
    return np.random.normal(mu, sigma, size=k)

def constant_initialization(k, c=0.):
    return np.full(k,c)

class bandit:
    def __init__(self, k, q, mode='s'):
        # Mode is 's' for stationary or 'ns' for non-stationary
        # Maybe we could pass an update function as parameter ?
        self.__k = k
        self.__q = q
        self.__ba = np.argmax(self.__q) # Best action
        self.__mode = mode

    def computeReward(self, A):
        # The action must be an integer
        # such that 0 <= A <= k - 1
        if (A < 0) or (A >= k): 
            return # Add exception
        reward = np.random.normal(loc=self.__q[A])
        self.__updateValues()
        self.__updateBestAction()
        return reward

    def getBestAction(self):
        return self.__ba

    def getNActions(self):
        return self.__k

    def __updateValues(self):
        if self.__mode == 's':
            return # Do nothing
        elif self.__mode == 'ns':
            self.__q += np.random.normal(scale=0.01, size=self.__k)
            return
        else:
            return #Add an exception

    def __updateBestAction(self):
        return np.argmax(self.__q)


class epsilon_greedy:
    def __init__(self, epsilon, n_step, bandit, debug=False):
        self.__epsilon = epsilon
        self.__k = bandit.getNActions()
        self.__n_step = n_step

        self.__Q = np.zeros(k)
        self.__N = np.zeros(k)

        self.__bandit = bandit

        self.__debug = debug

        if self.__debug:
            self.__opt_act_taken = []
            self.__list_rewards = []

    def __selectAction(self):
        is_greedy = np.random.uniform() > self.__epsilon
        if is_greedy:
            return self.__exploitation()
        else:
            return self.__exploration()

    def __exploitation(self):
        return np.argmax(self.__Q) # Would be better with random
    def __exploration(self):
        return np.random.randint(0, k)

    def __updateValues(self, action, reward):
        self.__N[action] += 1
        self.__Q[action] += (reward - self.__Q[action]) / self.__N[action]

    def getInfo(self):
        if self.__debug:
            return (self.__opt_act_taken, self.__list_rewards)
        else:
            return # Add exception

    def learn(self):
        for t in range(self.__n_step):
            action = self.__selectAction()
            reward = self.__bandit.computeReward(action)
            self.__updateValues(action, reward)

            if self.__debug:
                self.__opt_act_taken.append(action == self.__bandit.getBestAction())
                self.__list_rewards.append(reward)


if __name__ == "__main__":
    save = False

    k = 10
    n_run = 2000
    n_time = 1000

    epsilons = [0., 0.1, 0.01]
    results = {}

    oa = '% Optimal action'
    ar = 'Average reward'
    
    for epsilon in epsilons:
        print("Epsilon: ", epsilon)
        runs = {oa:[], ar:[]}
        for i in range(n_run):
            q = constant_initialization(k) # True values
            B = bandit(k, q, mode='ns')
            learner = epsilon_greedy(epsilon, n_time, B, debug=True)
            learner.learn()
            infos = learner.getInfo()

            runs[oa].append(infos[0])
            runs[ar].append(infos[1])

        runs[oa] = np.array(runs[oa]).mean(axis=0)*100
        runs[ar] = np.array(runs[ar]).mean(axis=0)*100

        results[epsilon] = runs
    
    fig, ax = plt.subplots()
    for epsilon in results:
        ax.plot(list(range(n_time)), results[epsilon][ar],
                label='$\epsilon = ' + str(epsilon) + '$', linewidth=1)
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
                label='$\epsilon = ' + str(epsilon) + '$', linewidth=1)
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
