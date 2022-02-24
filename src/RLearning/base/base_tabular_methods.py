from abc import ABC, abstractmethod
import numpy as np

class BaseTabularMethod(ABC):
  def __init__(self, envrioment, discount=1):
    """
    Tabular Methods base class

    Parameters
    ----------
    envrioment : Object that inherits from BaseEnvrioment
      Envrioment from where the method learns
    discount : float, default=1
      Discount factor for future rewards, should be in the interval [0, 1]
    """

    self.discount = discount
    self.envrioment = envrioment
    self.initialize_states()
    self.initialize_actions()
    self.initialize_policy()
    self.initialize_values()
  
  def initialize_states(self):
    """
    Store internally the envrioment states
    """
    self._states = self.envrioment.states
    self._num_states = len( self._states )

  def initialize_actions(self):
    """
    Store internally the envrioment actions
    """
    self._actions = self.envrioment.actions
    self._num_actions = len( self._actions )
  
  def initialize_policy(self):
    """
    Store internally the envrioment actions
    """
    self.policy = np.zeros( self._num_states, dtype=np.int32 )

  def initialize_values(self):
    """
    Initialize the State values and State-Action values.
    By default, all values are set to 0
    """
    self.state_value = np.zeros( self._num_states, dtype=np.float32 )
    self.state_action_value = np.zeros( (self._num_states, self._num_actions), dtype=np.float32 )

  @abstractmethod
  def action(self, state):
    """
    Return an action to the current state

    Parameters
    ----------
    state: int
      State ID representing the current state

    Returns
    -------
    int
      Action ID representing the action choosed
    """
    pass

  @abstractmethod
  def fit(self, episodes=10):
    """
    Optimize the model for the current envrioment.

    Parameters
    ----------
    episodes: int, default=10
      Number of episodes to simulate
    """
    pass

  @abstractmethod
  def simulate(self):
    """
    Simulate a single episode
    """
    pass

  @abstractmethod
  def policy_improvement(self):
    """
    Improve target policy
    """
    pass