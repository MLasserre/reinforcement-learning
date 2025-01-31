import numpy as np


# Functions initializing the true action values
def normal_initialization(k, mu=0., sigma=1.):
    """Initialize the true action values according to a normal distribution
    of parameters mu and sigma.

    Parameters
    ----------
    k : int
        Number of actions
    mu : float
        Mean of the normal distribution
    sigma : float
        Standard-deviation of the normal distribution

    Returns
    -------
    ndarray
        The true action values
    """
    return np.random.normal(mu, sigma, size=k)

def constant_initialization(k, c=0.):
    """Initialize the true action values to a same value.

    Parameters
    ----------
    k : int
        Number of actions
    c : float
        Value action
    
    Returns
    -------
    ndarray
        The true action values
    """
    return np.full(k,c)

def GRW_evolution(q, sigma):
    """ Function controlling the time evolution of the true action values
        in a non-stationary k-bandit. It defines a Gaussian Random Walk (GRW)
        of mean 0 and standard-deviation sigma.

    Parameters
    ----------
    q : float
        The current value of the action value.
    sigma : float
        The standard deviation of the GRW
    """
    return q + np.random.normal(scale=sigma, size=len(q))

class Bandit:
    def __init__(self, k):
        self.__k = k # Number of levers (actions)
        self.__ba = 0 # Best action
        self.__q = np.random.normal(0., 1., size=k) # True action values
        self.__updateBestAction()
        self.__ef = None

    def setValues(self, q):
        if len(Q) != self.__k:
            raise ValueError(f"Length mismatch: Q({len(Q)})"
                             f"must be of size {self.__k}.")
        self.__q = q
        self.__ba = self.__getBestAction()

    def setUpdateMethod(self, update_method):
        self.__ef = update_method

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
        self.__ba = np.argmax(self.__q)
