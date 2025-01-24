import numpy as np

class epsilon_greedy_learner:
    def __init__(self, bandit, epsilon=0.1, n_step=1000,
                 update_function=lambda N: N+1, debug=False):
        self.__epsilon = epsilon
        self.__k = bandit.getNActions()
        self.__n_step = n_step
        self.__uf = update_function

        self.__Q = np.zeros(self.__k)
        self.__N = np.zeros(self.__k)

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
        return np.random.randint(0, self.__k)

    def __updateValues(self, action, reward):
        self.__N[action] = self.__uf(self.__N[action])
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
