from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class BaseApproximatedMethod(ABC):
    def __init__(self, envrioment, 
                 featureExtractor,
                 controlValueApproximator, 
                 stateValueApproximator, 
                 discount=1):
        """
        Appriximated Methods base class

        Parameters
        ----------
        envrioment : Object inherits from BaseEnvrioment
            Envrioment from where the method learns
        featureExtractor : Sklearn-like transformer
            Method for extracting feature from the envrioment states 
        functionApproximator : Sklearn-like regressor
            Method for regressing values and action-values from feature extractors
        discount : int, default=1
            Discount factor for future rewards, should be in the interval [0, 1]
        """

        self.envrioment = envrioment
        self.featureExtractor = featureExtractor
        self.controlValueApproximator = controlValueApproximator
        self.stateValueApproximator = stateValueApproximator
        self.discount = discount

    @abstractmethod
    def action(self, state):
        pass

    @abstractmethod
    def fit(self, episodes=10):
        pass

    @abstractmethod
    def simulate(self):
        pass

    @abstractmethod
    def policy_improvement(self):
        pass

    @abstractmethod
    def state_value(self, state):
        pass

    @abstractmethod
    def state_action_value(self, state, action):
        pass


