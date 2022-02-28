import numpy as np
import random

from RLearning.base.base_methods import BaseMethod
from RLearning.interfaces import TabularInterface

INFINITY = np.inf

class MonteCarlo( BaseMethod ):
  def __init__(self, *args, env_interface=TabularInterface(), eps=0.0, **kwargs):
    """Monte Carlo

    Parameters
    ----------
    env_interface : interface, optional
        Interface between the agent and the envrioment, by default TabularInterface()
    eps : float, optional
        Eps probability for eps-greedy policy, by default 0.0
    """
    self._eps = eps
    self.env_interface = env_interface
    super(MonteCarlo, self).__init__(*args, **kwargs)

  def action( self, state ):
    if np.random.uniform( 0, 1 ) < self._eps:
        return self.env_interface.choose_random_action()

    return self.env_interface.choose_greedy_action(state) 

  def fit(self, envrioment):
    self.env_interface.fit(envrioment)

    for episode in range( self.episodes ):
      self.env_interface.initialize_envrioment()
      self.simulate()

  def simulate(self):
    actions = []
    states  = []
    rewards = []

    ## Playing the game
    time = 0
    while not self.env_interface.is_terminal():
      state = self.env_interface.state()
      action = self.action(state)
      reward = self.env_interface.reward(action)

      rewards.append( reward )
      states.append( state )
      actions.append( action )

      time+=1

    ## Policy Improvement
    self.policy_improvement( states, rewards, actions )

  def policy_improvement(self, states, rewards, actions):
    ## Policy evaluation
    cumulative_return = 0.0
    for time_back in reversed( range(0, len(rewards)) ):
      state = states[time_back]
      reward = rewards[time_back]
      action = actions[time_back]
      cumulative_return = cumulative_return*self.discount + reward

      self.env_interface.update_state_value( state, cumulative_return )
      self.env_interface.update_control_value( state, action, cumulative_return )

      time_back-=1

"""
DEPRECATED
class MonteCarlo(BaseTabularMethod):
  def __init__(self, *args, eps=0.0, **kwargs):
    self._eps = eps
    super(MonteCarlo, self).__init__(*args, **kwargs)
  
  def action( self, state ):
    if np.random.uniform( 0, 1 ) < self._eps:
        return np.random.randint(0, self._num_actions)
    return self.policy[ state ]

  def fit( self, episodes=10 ):
    self._initialize_fit_variables()

    for episode in range(episodes):
      self.envrioment.initialize()
      self.simulate()

    return self

  def _initialize_fit_variables(self):
    # Initalizate state action counts
    # Used for update the mean
    self.state_action_count = np.zeros( (self._num_states, self._num_actions) ) 
    self.state_count = np.zeros( self._num_states ) 

    # Store first vist position
    self._first_visit_state = np.zeros( self._num_states, dtype=np.int )
    self._first_visit_state[:] = INFINITY 

    self._first_vist_action_state = np.zeros( (self._num_states, self._num_actions), dtype=np.int )
    self._first_vist_action_state[:,:] = INFINITY

  def simulate(self):
    actions = []
    states  = []
    rewards = []

    ## Playing the game
    time = 0
    while not self.envrioment.is_terminal():
      state = self.envrioment.state()
      action = self.action(state)
      reward = self.envrioment.reward(action)

      rewards.append( reward )
      states.append( state )
      actions.append( action )

      self._first_visit_state[state] = min( self._first_visit_state[state], time )
      self._first_vist_action_state[state][action] = min( self._first_vist_action_state[state][action], time )

      time+=1

    ## Policy Improvement
    self.policy_improvement( states, rewards, actions )

  def policy_improvement(self, states, rewards, actions):
    ## Policy evaluation
    cumulative_return = 0.0
    for time_back in reversed( range(0, len(rewards)) ):
      state = states[time_back]
      reward = rewards[time_back]
      action = actions[time_back]
      cumulative_return = cumulative_return*self.discount + reward
      
      # First-visit Monte Carlo
      # Update state value
      if time_back == self._first_visit_state[state]:
        self.state_count[state] += 1
        self.state_value[state] += (cumulative_return - self.state_value[state])/self.state_count[state]
        
        # Reset first vist
        self._first_visit_state[state] = INFINITY
        
      # Update state-action value
      if time_back == self._first_vist_action_state[state][action]:
        self.state_action_count[state][action] += 1
        self.state_action_value[state][action] += (cumulative_return - self.state_action_value[state][action] )/self.state_action_count[state][action]
        
        # Reset first vist
        self._first_vist_action_state[state][action] = INFINITY

        ## Policy improvement
        self._improve_state_action_policy(state, action)

      time_back-=1

  def _improve_state_action_policy(self, state, action):
    best_action = self.policy[state]
    if self.state_action_value[state][action] > self.state_action_value[state][best_action]:
      self.policy[state]=action

class MonteCarloApproximated(BaseApproximatedMethod):
  def action(self, state):
    # Random action
    if np.random.uniform() < self.eps:
      return self._choose_random_action()

    try:
      return self._choose_greedy_action(state)
    except:
      return self._choose_random_action()

  def fit(self, episodes=10):

    for episode in range( episodes ):
      self.envrioment.initialize()
      self.simulate()
  
  def simulate(self):
    actions = []
    states = []
    rewards = []

    time = 0
    while not self.envrioment.is_terminal():
      state = self.envrioment.state()
      action = self.action(state)
      reward = self.envrioment.reward(action)

      rewards.append( reward )
      states .append( state  )
      actions.append( action )
      time+=1
    
    self.policy_improvement(rewards, states, actions)

  def policy_improvement(self, rewards, states, actions):
    cumulative_return = 0.0
    for time_back in reversed( range(0, len(rewards)) ):
      state = states[time_back]
      reward = rewards[time_back]
      action = actions[time_back]
      cumulative_return = cumulative_return*self.discount + reward
      
      self._update_state_values( state, cumulative_return )
      self._update_control_values( state, action, cumulative_return )
      
  def _update_state_values(self, state, value):
    if self.state_feature_extractor==None or self.state_value_approximator==None:
      return

    state_features = self.state_feature_extractor.transform( [state] )
    self.state_value_approximator.partial_fit( X=state_features, y=[value] )
  
  def _update_control_values(self, state, action, value):
    
    control_pair = np.hstack( (np.array(state), np.array(action)) )
    control_features = self.control_feature_extractor.transform( [control_pair] )
    
    self.control_value_approximator.partial_fit( X=control_features, y=[value] )

  def state_value(self, state):
    state_features = self.state_feature_extractor.transform( [state] )
    return self.state_value_approximator.predict( state_features )[0]
  
  def state_action_value(self, state, action):
    control_pair = np.hstack( [np.array(state), np.array(action)] )
    control_features = self.control_feature_extractor.transform( [control_pair] )

    return self.control_value_approximator.predict( control_features )
"""