import numpy as np
from collections import defaultdict

class TabularInterface():
    def __init__( self, alpha=0.1 ):
        """Tabular interface for tabular methods.

        Parameters
        ----------
        alpha : float, optional
            step-size parameter, by default 0.1
        """
        self.alpha = alpha
        
    def fit( self, envrioment ):
        self.envrioment = envrioment

        self._initialize_actions()
        self._initialize_values()
        self._initializate_values()

        return self
    
    # Control Methods
    def _initialize_actions(self):
        self._actions = self.envrioment.actions
        self._id_to_action = dict( enumerate( self._actions ) )
        self._action_to_id = { action:id for id,action in self._id_to_action.items() }
    
    def _initialize_values(self):
        self._states = self.envrioment.states
        self._id_to_state = dict( enumerate( self._states ) )
        self._state_to_id = { state:id for id,state in self._id_to_state.items() }

    def _initializate_values(self):
        self.state_value = np.zeros( len(self._states) )
        self.state_action_value = np.zeros( (len(self._states), len(self._actions)) )

    def set_initial_values(self, value):
        self._initializate_values()
        self.state_action_value[ :, : ] = value

    def get_state_value( self, state_id ):
        return self.state_value[state_id]

    def get_control_value( self, state_id, action_id ):
        return self.state_action_value[state_id, action_id]

    def get_state_action_values( self, state_id ):
        return self.state_action_value[state_id, :]

    def get_states_values( self ):
        return self.state_value

    def update_state_value( self, state_id, target ):
        difference = target-self.state_value[state_id]

        self.state_value[state_id] += difference*self._step_size()
    
    def update_control_value( self, state_id, action_id, target ):
        difference = target-self.state_action_value[state_id][action_id]

        self.state_action_value[state_id][action_id] += difference*self._step_size()

    def _step_size(self):
        return self.alpha

    def choose_random_action(self):
        return np.random.randint( 0, len( self._actions ) )

    def choose_greedy_action(self, state_id):
        action_values = self.state_action_value[ state_id, : ]
        best_action_id = np.argmax( action_values )
        best_action_id = int( best_action_id )

        return best_action_id
    
    # Envrioment Interaction Methods
    def initialize_envrioment(self):
        self.envrioment.initialize()

    def state(self):
        state = self.envrioment.state()
        state_id = self._state_to_id[ state ]

        return state_id

    def reward(self, action_id):
        action = self._id_to_action[ action_id ]
        reward = self.envrioment.reward(action)

        return reward

    def is_terminal(self):
        return self.envrioment.is_terminal()
        