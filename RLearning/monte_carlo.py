import numpy as np
import random

from collections import defaultdict

from RLearning.base.base_methods import BaseMethod
from RLearning.interfaces import TabularInterface

INFINITY = np.inf

class MonteCarlo( BaseMethod ):
  def __init__(self, *args, env_interface=TabularInterface(), eps=0.0, mode='first-visit', **kwargs):
    """Monte Carlo

    Parameters
    ----------
    env_interface : interface, optional
        Interface between the agent and the envrioment, by default TabularInterface()
    eps : float, optional
        Eps probability for eps-greedy policy, by default 0.0
    mode : str, optional
        The algorithm behaviour when updating the values, should be 'first-visit' or 'any-visit'.
        The 'first-visit' mode only works with Tabular Envrioment.
    """

    self._eps = eps
    self.env_interface = env_interface
    self.mode = mode
    super(MonteCarlo, self).__init__(*args, **kwargs)

  def action( self, state ):
    if np.random.uniform( 0, 1 ) < self._eps:
        return self.env_interface.choose_random_action()

    return self.env_interface.choose_greedy_action(state) 

  def fit(self, envrioment):
    self.env_interface.fit(envrioment)
    self._create_first_visit_variables()

    for episode in range( self.episodes ):
      self.env_interface.initialize_envrioment()
      self.simulate()

  def _create_first_visit_variables(self):
    self.state_first_vist = defaultdict( lambda: np.inf )
    self.control_first_vist = defaultdict( lambda: np.inf )

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

      if self.mode == 'first-visit':
        self.state_first_vist[state] = min( self.state_first_vist[state], time )
        self.control_first_vist[(state, action)] = min( self.control_first_vist[(state, action)], time )

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

      if self._can_improve(time_back, state):
        self.env_interface.update_state_value( state, cumulative_return )

      if self._can_improve(time_back, state, action):
        self.env_interface.update_control_value( state, action, cumulative_return )

      time_back-=1

  def _can_improve( self, time, state, action=None ):
    if self.mode != 'first-visit':
      return True
    
    if action == None:
      return_value = self.state_first_vist[state] == time
      self.state_first_vist[state] = np.inf

      return return_value

    return_value = self.control_first_vist[ (state, action) ] == time
    self.control_first_vist[(state, action)] = np.inf
    
    return return_value
