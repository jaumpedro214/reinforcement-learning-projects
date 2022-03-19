from RLearning.base.base_methods import BaseMethod
from RLearning.interfaces import TabularInterface

import collections

import numpy as np

class SARSA(BaseMethod):
  def __init__(self, *args, env_interface=TabularInterface(), eps=0.0, **kwargs):
    """SARSA

    Parameters
    ----------
    env_interface : interface, optional
        Interface between the agent and the environment, by default TabularInterface()
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

  def fit(self, environment):
    self.env_interface.fit( environment )

    for episode in range( self.episodes ):
      self.env_interface.initialize_environment()
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
    probability = self._eps/len(self.env_interface._actions)
    if self.env_interface.choose_greedy_action(state) == action:
      probability += 1-self._eps
    return probability

class NStepSarsa(BaseMethod):
  def __init__(self, *args, env_interface=TabularInterface(), eps=0.0, n_steps=1, off_policy=False, **kwargs):
    """N-Step SARSA

    Parameters
    ----------
    env_interface : interface, optional
        Interface between the agent and the environment, by default TabularInterface()
    eps : float, optional
        Eps probability for eps-greedy policy, by default 0.0
    n_steps : int, optional
        Number of steps ahead the current step to consider when updating the state value, by default 1.
        Should be greater or equal to 1
    off_policy : bool, optional
        Whether or not to use off-policy. 
        If True, the behavior policy is the eps-greedy, while the target policy is deterministic.
    """
    self._eps = eps
    self.env_interface = env_interface
    self.n_steps = n_steps
    self.off_policy = off_policy
    super(NStepSarsa, self).__init__(*args, **kwargs)
  
  def action( self, state ):
    if np.random.uniform( 0, 1 ) < self._eps:
        return self.env_interface.choose_random_action()
    return self.env_interface.choose_greedy_action(state) 
  
  def fit(self, environment):
    self.env_interface.fit( environment )
    self.discounts = np.array( [self.discount**i for i in range(0, self.n_steps+1)], dtype=float )
    
    for episode in range( self.episodes ):
      self.env_interface.initialize_environment()
      self.simulate()

  def simulate(self):
    rewards = np.zeros( self.n_steps+1, dtype=float )
    states =  [ None ]*( self.n_steps+1 )
    actions = [ None ]*( self.n_steps+1 )
    importance_sampling = [ None ]*( self.n_steps+1 )
    time = 0

    while not self.env_interface.is_terminal():
      state = self.env_interface.state()
      action = self.action(state)
      reward = self.env_interface.reward(action)
      relative_probability = self.relative_probability(state, action)

      if time <= self.n_steps:
        states[time] = state 
        actions[time] = action
        rewards[time] = reward
        importance_sampling[time] = relative_probability
        time+=1
        continue

      self.state_value_update( states, rewards, importance_sampling )
      self.control_value_update( states, actions, rewards, importance_sampling )
      rewards[:-1] = rewards[1:]
      rewards[-1] = reward

      states[:-1] = states[1:]
      states[-1] = state

      actions[:-1] = actions[1:]
      actions[-1] = action

      importance_sampling[:-1] = importance_sampling[1:]
      importance_sampling[-1] = relative_probability

      time+=1

    # Final updates after the episode's ending
    while states[0] != None:
      self.state_value_update( states, rewards, importance_sampling )
      self.control_value_update( states, actions, rewards, importance_sampling )
      rewards[:-1] = rewards[1:]
      rewards[-1] = 0
      states[:-1] = states[1:]
      states[-1] = None
      actions[:-1] = actions[1:]
      actions[-1] = None
      importance_sampling[:-1] = importance_sampling[1:]
      importance_sampling[-1] = None

  def state_value_update(self, states, rewards, samplings):
    if self.off_policy:
      self.off_state_value_update( states, rewards, samplings )
      return
    self.on_state_value_update( states[0], rewards, states[-1] )

  def control_value_update(self, states, actions, rewards, samplings):
    if self.off_policy:
      self.off_control_value_update( states, actions, rewards, samplings)
      return
    self.on_control_value_update( states[0], actions[0], rewards, states[-1], actions[-1] )

  def off_state_value_update(self, states, rewards, samplings):
    
    target_G = 0
    start_sum = False

    state_target = states[0]
    rewards = list( rewards )

    for state, reward, p_sampling, time in reversed(list( zip(states, rewards, samplings, range(0, len(states)) ) )):
      if state == None:
        continue

      if not start_sum:
        # If the episode has not ended 
        target_G = self.env_interface.get_state_value(state)
        start_sum = True
        continue

      new_target_G = 0
      new_target_G += p_sampling*(reward + self.discount*target_G)
      new_target_G += (1-p_sampling)*self.env_interface.get_state_value(state)

      target_G = new_target_G

    self.env_interface.update_state_value(state_target, target_G)

  def off_control_value_update(self, states, actions, rewards, samplings):
    target_G = 0
    start_sum = False

    state_target = states[0]
    action_target = actions[0]

    for state, action, reward, p_sampling, time in reversed(list(zip(states, actions, rewards, samplings, range(0, len(states))))):
      if state == None:
        continue
      
      if not start_sum:
        # If the episode has not ended 
        target_G = self.env_interface.get_control_value(state, action)
        
        # If the episode has ended
        if time != len(states)-1:
          target_G = reward
        start_sum = True
        continue
      
      new_target_G = 0
      new_target_G += reward 
      new_target_G += self.discount*p_sampling*(target_G - self.env_interface.get_control_value(state, action))
      new_target_G += self.discount*self._expected_reward_state( state )

      target_G = new_target_G

    self.env_interface.update_control_value( state_target, action_target, target_G )

  def on_state_value_update(self, state, rewards, last_state=None):
    rewards_ = rewards.copy()
    if last_state == None:
      rewards_[-1] = 0
    else:
      rewards_[-1] = self.env_interface.get_state_value( last_state )

    target = (self.discounts*rewards_).sum()
    self.env_interface.update_state_value( state, target )

  def on_control_value_update(self, state, action, rewards, last_state=None, last_action=None):
    rewards_ = rewards.copy()
    if last_state == None:
      rewards_[-1] = 0
    else:
      rewards_[-1] = self.env_interface.get_control_value( last_state, last_action )

    target = (self.discounts*rewards_).sum()
    self.env_interface.update_control_value( state, action, target )

  ## Off-Policy auxiliary methods
  def relative_probability( self, state, action ):
    if self.off_policy:
      return self.target_probability(state,action)/self.behavior_probability(state,action)
    return 1.0

  def target_probability( self, state, action ):
    if action==self.env_interface.choose_greedy_action(state):
      return 1.0
    return 0.0

  def behavior_probability( self, state, action ):
    # Avoid inconsistencies
    if self._eps <= 0.0:
      return 1.0

    # The behavior policy is eps greedy
    probability = self._eps/len(self.env_interface._actions)
    if action == self.env_interface.choose_greedy_action(state):
      probability += 1-self._eps
    return probability

  def _expected_reward_state( self, state ):
    probas = np.array( [ self.behavior_probability(state, action) for action in range( len(self.env_interface._actions) ) ] )
    rewards = np.array( [ self.env_interface.get_control_value(state, action) for action in range( len(self.env_interface._actions) ) ] )
    return np.sum( probas*rewards )