from abc import ABC, abstractmethod
import numpy as np

class BaseTabularEnvrioment(ABC):
  def __init__(self):
    self.initialize_states()
    self.initialize_actions()

  @abstractmethod
  def initialize(self):
    pass

  @abstractmethod
  def initialize_states(self):
    pass

  @abstractmethod
  def initialize_actions(self):
    pass

  @abstractmethod
  def state(self):
    pass

  @abstractmethod
  def reward(self, action):
    pass

class BaseTabularMethod(ABC):
  def __init__(self, envrioment, discount=1):
    self.discount = discount
    self.envrioment = envrioment
    self.initialize_states()
    self.initialize_actions()
    self.initialize_policy()
    self.initialize_values()
  
  def initialize_states(self):
    self._states = self.envrioment.states
    self._num_states = len( self._states )

  def initialize_actions(self):
    self._actions = self.envrioment.actions
    self._num_actions = len( self._actions )
  
  def initialize_policy(self):
    # Random Initial Policy
    self.policy = np.zeros( self._num_states, dtype=np.int32 )

  def initialize_values(self):
    self.state_value = np.zeros( self._num_states, dtype=np.float32 )
    self.state_action_value = np.zeros( (self._num_states, self._num_actions), dtype=np.float32 )

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