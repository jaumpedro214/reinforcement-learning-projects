from RLearning.base.base_envrioment import BaseEnvrioment
import random
import itertools
import numpy as np

class RandomDiscreteWalk(BaseEnvrioment):
  def initialize(self):
    self._position = 2 # Starts on 'C'
    self._reach_terminal = False

  def initialize_states(self):
    self.states = ['A', 'B', 'C', 'D', 'E']
    self.terminal_state = 'T'
    self.states.append( self.terminal_state )

    self._id_to_state = dict( enumerate(self.states) ) 
    self._state_to_id = { state:id for id,state in self._id_to_state.items() }

  def initialize_actions(self):
    self.actions = ['NONE']
    self._id_to_action = dict( enumerate(self.actions) ) 
    self._action_to_id = { action:id for id,action in self._id_to_action.items() }

  def state(self):
    if self._position < 0 or self._position >= len(self.states)-1:
      return self._state_to_id[ self.terminal_state ]

    state = self.states[ self._position ]
    state_id  = self._state_to_id[ state ]
    return state_id

  def update_state(self):
    self._position += random.choice( [-1,1] )

  def is_terminal(self):
    if self._reach_terminal:
      return True

    self._reach_terminal = self._position<0 or self._position+1>=len(self.states)
    return self._reach_terminal
  
  def reward(self, action):
    if self.is_terminal():
      return 0

    self.update_state()
    if self._position < 0:
      return 0
    if self._position >= len(self.states)-1:
      return 1

    return 0

class Random1000StateWalk(BaseEnvrioment):
  def initialize(self):
    self._position = 500 # Starts on 'C'
    self._reach_terminal = False

  def initialize_states(self):
    pass

  def initialize_actions(self):
    self.actions = [0]
  
  def state(self):
    return self._position

  def update_state(self):
    self._position += np.random.randint(-100, 100+1)

  def is_terminal(self):
    if self._reach_terminal:
      return True

    self._reach_terminal = self._position<0 or self._position>=1000
    return self._reach_terminal

  def reward(self, action):
    if self.is_terminal():
      return 0

    self.update_state()
    if self._position < 1:
      return -1
    if self._position > 1000:
      return 1

    return 0
  
class WindyGridWorld(BaseEnvrioment):
  def initialize(self):
    self._player_position = self._start_point
    self._player_in_game = True

  def initialize_states(self):
    self._n_lines = 7
    self._n_columns = 10

    self._start_point = (3, 0)
    self._end_point = (3, 7)
    
    self._columns_wind = [0,0,0,1,1,1,2,2,1,0]

    self.states = list( itertools.product( range(self._n_lines), range(self._n_columns) ) )
    self.terminal_state = (-1, -1)
    self.states.append( self.terminal_state )
    self._state_to_id = { state:id for id,state in enumerate(self.states) }

  def initialize_actions(self):
    self.actions = [(+1, 0), (-1, 0), (0,+1), (0,-1)]
    self._action_to_id = { action:id for id,action in enumerate(self.actions) }

  def state(self):
    if not self._player_in_game:
      return self._state_to_id[ self.terminal_state ]

    return self._state_to_id[ self._player_position ]

  def _wind(self, column ):
    return self._columns_wind[column]
  
  def reward(self, action_id):
    if not self._player_in_game:
      return 0

    action = self.actions[ action_id ]

    lin_pos = self._player_position[0]
    col_pos = self._player_position[1]

    col_pos = max(0, min(col_pos+action[1],self._n_columns-1))
    lin_pos = max(0, min(lin_pos+action[0]+self._wind(col_pos), self._n_lines-1))

    if self._player_position == self._end_point:
      self._player_in_game = False
      return 1

    self._player_position = ( lin_pos, col_pos )

    return -1

  def is_terminal(self):
    return not self._player_in_game

class MontainCar(BaseEnvrioment):
  def initialize(self):
    self._position = np.random.uniform( -0.6, -0.4 )
    self._velocity = 0
    self._player_in_game = True

  def initialize_actions(self):
    # Full throttle, Reverse throttle, zero 
    self.actions = [+1, -1, 0]

  def initialize_states(self):
    pass

  def state(self):
    state = [ self._position, self._velocity ]
    return state

  def reward(self, action_id):
    action = self.actions[action_id]
    
    r_reward = -1
    if self._position >= 0.5:
      self._player_in_game = False
      r_reward = 1
    
    self._update_state(action)
    return r_reward

  def _update_state(self, action):
    self._velocity = self._velocity + 0.001*action - 0.0025*np.cos( 3*self._position )
    self._velocity = max( min( self._velocity, 0.07 ), -0.07 )

    self._position = self._position+self._velocity
    self._position = max( min( self._position, 0.51 ), -1.200001 )

    if self._position <= -1.2:
      self._velocity = 0

  def is_terminal(self):
    return not self._player_in_game