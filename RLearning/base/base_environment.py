from abc import ABC, abstractmethod
import numpy as np

class BaseEnvironment(ABC):
  """
  Base environment class
  """
  def __init__(self):
    self.initialize_states()
    self.initialize_actions()

  @abstractmethod
  def initialize(self):
    """
    Initialize the environment variables to simulate a new episode
    """
    pass

  @abstractmethod
  def initialize_states(self):
    """
    If tabular, initialize all possible states
    If approximated, initialize environment's feature construction
    """
    pass

  @abstractmethod
  def initialize_actions(self):
    """
    Initialize all possible actions
    """
    pass

  @abstractmethod
  def state(self):
    """
    Return the current state
    """
    pass

  @abstractmethod
  def reward(self, action_id):
    """
    Recieves an action ID (that is transformed in a real action
    following internal rules) and return the actual reward.

    Parameters
    ----------
    action_id : int
        Numerical ID relative to the action

    Returns
    -------
    reward : int or float
        Current state reward 
    """
    pass
