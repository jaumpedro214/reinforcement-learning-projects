from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class BaseApproximatedMethod(ABC):
    def __init__(self, envrioment, 
                 control_feature_extractor,
                 control_value_approximator,
                 state_feature_extractor=None, 
                 state_value_approximator=None,
                 discount=1,
                 eps=0.0):
        """
        Appriximated Methods base class

        Parameters
        ----------
        envrioment : Object inherits from BaseEnvrioment
            Envrioment from where the method learns
        control_feature_extractor : Sklearn-like transformer
            Method for extracting features from the envrioment state-action pairs
        control_value_approximator : Sklearn-like regressor
            Method for regressing action-state values
        state_feature_extractor : Sklearn-like transformer, optional
            Method for extracting features from the envrioment states
        state_value_approximator : Sklearn-like regressor
            Method for regressing state values
        discount : int, default=1
            Discount factor for future rewards, should be in the interval [0, 1]
        eps : foat, default=0.0
            Epsilon value for e-greedy policy
        """

        self.envrioment = envrioment
        self.control_feature_extractor = control_feature_extractor
        self.control_value_approximator = control_value_approximator
        self.state_feature_extractor = state_feature_extractor
        self.state_value_approximator = state_value_approximator
        self.discount = discount
        self.eps = eps

        self.initialize_actions()
    
    def initialize_actions(self):
        """
        Store internally the envrioment actions
        """
        self._actions = self.envrioment.actions
        self._num_actions = len( self._actions )

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


