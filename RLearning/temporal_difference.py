from RLearning.base.base_methods import BaseMethod
from RLearning.interfaces import TabularInterface

import numpy as np

class SARSA(BaseMethod):
  def __init__(self, *args, env_interface=TabularInterface(), eps=0.0, **kwargs):
    """SARSA

    Parameters
    ----------
    env_interface : interface, optional
        Interface between the agent and the envrioment, by default TabularInterface()
    eps : float, optional
        Eps probability for eps-greedy policy, by default 0.0
    """
    self._eps = eps
    self.env_interface = env_interface
    super(SARSA, self).__init__(*args, **kwargs)

  def action( self, state ):
    if np.random.uniform( 0, 1 ) < self._eps:
        return self.env_interface.choose_random_action()
    return self.env_interface.choose_greedy_action(state) 

  def fit(self, envrioment):
    self.env_interface.fit( envrioment )

    for episode in range( self.episodes ):
      self.env_interface.initialize_envrioment()
      self.simulate()

  def simulate(self):
    
    current_state = self.env_interface.state()
    current_action = self.action(current_state)

    while not self.env_interface.is_terminal():
      reward = self.env_interface.reward(current_action)

      next_state = self.env_interface.state()
      next_action = self.action(next_state)

      self.state_value_update(reward, current_state, next_state)
      self.control_value_update(current_state, current_action, reward, next_state, next_action)

      current_state=next_state
      current_action=next_action

  def state_value_update(self, reward, current_state, next_state):
    target = reward + self.discount*self.env_interface.get_state_value( next_state )
    self.env_interface.update_state_value( current_state, target )
  
  def control_value_update( self, current_state, current_action, reward, next_state, next_action ):
    target = reward
    target += self.discount*self.env_interface.get_control_value( next_state, next_action )
    self.env_interface.update_control_value( current_state, current_action, target )

class QLearning(SARSA):
  def __init__(self, *args, **kwargs):
    super(QLearning, self).__init__(*args, **kwargs)

  def simulate(self):
    current_state = self.env_interface.state()
    while not self.env_interface.is_terminal():
      action = self.action(current_state)
      reward = self.env_interface.reward(action)
      next_state = self.env_interface.state()

      self.state_value_update(reward, current_state, next_state)
      self.state_action_value_update(current_state, action, reward, next_state)
      current_state=next_state

  def state_value_update(self, reward, current_state, next_state):
    target = reward+self.discount*np.max( self.env_interface.get_states_values() )
    self.env_interface.update_state_value(current_state, target)

  def state_action_value_update(self, current_state, action, reward, next_state):
    target = reward
    target += self.discount*np.max( self.env_interface.get_state_action_values(next_state) )

    self.env_interface.update_control_value( current_state, action, target )

class ExpectedSARSA(SARSA):
  def __init__(self, *args, **kwargs):
    """
    Expected version of SARSA. 
    Estimates control (S,a) target value considering the expected value of a stochastic eps-greedy policy.
    
    """
    super(ExpectedSARSA, self).__init__(*args, **kwargs)
  
  def control_value_update( self, current_state, current_action, reward, next_state, next_action ):
    target = reward
    target += self.discount*self._expected_reward_state(next_state)
    self.env_interface.update_control_value( current_state, current_action, target )

  def _expected_reward_state( self, state ):
    probas = np.array( [ self._state_action_proba(state, action) for action in range( len(self.env_interface._actions) ) ] )
    rewards = np.array( [ self.env_interface.get_control_value(state, action) for action in range( len(self.env_interface._actions) ) ] )

    return np.sum( probas*rewards )

  def _state_action_proba(self, state, action):
    # This class is eps-greedy
    # p -> eps/|actions| for non-optimal actions
    # p -> 1-eps + eps/|actions| for the optimal action
    p = self._eps/len(self.env_interface._actions)
    if self.env_interface.choose_greedy_action(state) == action:
      p += 1-self._eps
    return p

class NStepSarsa(BaseMethod):
  def __init__(self, *args, env_interface=TabularInterface(), eps=0.0, n_steps=1, **kwargs):
    """N-Step SARSA

    Parameters
    ----------
    env_interface : interface, optional
        Interface between the agent and the envrioment, by default TabularInterface()
    eps : float, optional
        Eps probability for eps-greedy policy, by default 0.0
    n_steps : int, optional
        Number of steps ahead the current step to consider when updating the state value, by default 1.
        Should be greater or equal to 1
    """
    self._eps = eps
    self.env_interface = env_interface
    self.n_steps = n_steps
    super(NStepSarsa, self).__init__(*args, **kwargs)
  
  def action( self, state ):
    if np.random.uniform( 0, 1 ) < self._eps:
        return self.env_interface.choose_random_action()
    return self.env_interface.choose_greedy_action(state) 
  
  def fit(self, envrioment):
    self.env_interface.fit( envrioment )

    for episode in range( self.episodes ):
      self.env_interface.initialize_envrioment()
      self.simulate()

  def simulate(self):
    discounts = np.array( [self.discount**i for i in range(0, self.n_steps+1)] )
    rewards = np.zeros( self.n_steps+1 )
    states = []
    actions = []
    time = 0

    while not self.env_interface.is_terminal():
      state = self.env_interface.state()
      action = self.action(state)
      reward = self.env_interface.reward(action)

      if time <= self.n_steps:
        states.append( state )
        actions.append( action )
        rewards[time]=reward
        continue
      
      self.state_value_update( states[0], rewards, discounts, last_state=states[-1] )
      self.control_value_update( states[0], actions[0], rewards, discounts, last_state=states[-1], last_action=actions[-1] )
      rewards[:-1] = rewards[1:]
      rewards[-1] = reward

      states[:-1] = states[1:]
      states[-1] = state

      actions[:-1] = actions[1:]
      actions[-1] = action

      time+=1

    # Final updates after the episode's ending
    while states[0] != None:
      self.state_value_update( states[0], rewards, discounts, last_state=states[-1] )
      self.control_value_update( states[0], actions[0], rewards, discounts, last_state=states[-1], last_action=actions[-1] )
      rewards[:-1] = rewards[1:]
      rewards[-1] = 0
      states[:-1] = states[1:]
      states[-1] = None
      actions[:-1] = actions[1:]
      actions[-1] = None

  def state_value_update(self, state, rewards, discounts, last_state=None):
    rewards[-1] = self.env_interface.get_state_value( last_state ) if last_state != None else 0
    target = np.sum(discounts*rewards)

    self.env_interface.update_state_value( state, target )

  def control_value_update(self, state, action, rewards, discounts, last_state=None, last_action=None):
    rewards[-1] = self.env_interface.get_control_value( last_state, last_action ) if last_state != None else 0
    target = np.sum(discounts*rewards)

    self.env_interface.update_control_value( state, action, target )
