import numpy as np


# Functions initializing the true action values
def normal_initialization(k, mu=0., sigma=1.):
    return np.random.normal(mu, sigma, size=k)
def constant_initialization(k, c=0.):
    return np.full(k,c)

# Function controlling the time evolution of true action values
# in a non-stationary k-bandit
def GRW_evolution(q, sigma):
    return q + np.random.normal(scale=sigma, size=len(q))

class bandit:
    def __init__(self, k, q, evo_func=None):
        self.__k = k # Number of levers (actions)
        self.__q = q # True action values
        self.__ba = np.argmax(self.__q) # Best action
        self.__ef = evo_func

    def computeReward(self, a):
        # The action must be an integer
        # such that 0 <= a <= k - 1
        if (a < 0) or (a >= self.__k): 
            return # Add exception
        reward = np.random.normal(loc=self.__q[a])
        if self.__ef:
            self.__q = self.__ef(self.__q)
            self.__updateBestAction()
        return reward

    def getBestAction(self):
        return self.__ba

    def getNActions(self):
        return self.__k

    def __updateBestAction(self):
        return np.argmax(self.__q)



