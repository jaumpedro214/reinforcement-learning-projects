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
        self._initialize_states()
        self._initialize_values()

        return self
    
    # Control Methods
    def _initialize_actions(self):
        self._actions = self.envrioment.actions
        self._id_to_action = dict( enumerate( self._actions ) )
        self._action_to_id = { action:id for id,action in self._id_to_action.items() }
    
    def _initialize_states(self):
        self._states = self.envrioment.states
        self._id_to_state = dict( enumerate( self._states ) )
        self._state_to_id = { state:id for id,state in self._id_to_state.items() }

    def _initialize_values(self):
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

class ApproximatedInterface():
    def __init__(self, 
                 control_feature_extractor, 
                 control_value_approximator,
                 state_feature_extractor=None,
                 state_value_approximator=None
                ):
        
        self.control_feature_extractor = control_feature_extractor
        self.control_value_approximator = control_value_approximator
        self.state_feature_extractor = state_feature_extractor
        self.state_value_approximator = state_value_approximator

    def fit(self, envrioment ):
        self.envrioment = envrioment

        self._initialize_actions()
        self._fit_models()
        return self

    def _approximating_value(self):
        if self.state_value_approximator == None:
            return False
        if self.state_feature_extractor == None:
            return False
        return True

    def _initialize_actions(self):
        self._actions = self.envrioment.actions
        self._id_to_action = dict( enumerate( self._actions ) )
        self._action_to_id = { action:id for id,action in self._id_to_action.items() }
    
    def _fit_models(self):
        self.initialize_envrioment()

        state = self.envrioment.state()
        action = self.choose_random_action()

        if self._approximating_value():
            self.state_feature_extractor.fit( [state] )

        self.control_feature_extractor.fit( np.hstack( (np.array(state), np.array(action)) ) )

        self.update_state_value( state, 0 )
        self.update_control_value( state, action, 0 )

        self.initialize_envrioment()

    ## Control Methods
    def get_state_value(self, state):
        if not self._approximating_value():
            return 0

        state_vector = self.state_feature_extractor.transform( [state_vector] )

        try:
            return self.state_value_approximator.predict( state_vector )[0]
        except:
            return 0

    def get_control_value( self, state, action ):
        control = np.hstack( (np.array(state), np.array(action)) )
        control_vector = self.control_feature_extractor.transform( [control] )

        try:
            return self.control_value_approximator.predict( control_vector )[0]
        except:
            return 0

    def get_state_action_values( self, state ):
        control_values = [ self.get_control_value(state, action) for action in self._actions ]
        return np.array(control_values)
    
    def get_states_values( self ):
        """NOT AVAILABLE FOR APP METHODS

        Returns
        -------
        [0]
        """
        return [0]

    def update_state_value( self, state, target ):
        if not self._approximating_value():
            return

        state_vector = self.state_feature_extractor.transform( [state] )
        self.state_value_approximator.partial_fit( X=state_vector, y=[target] )
    
    def update_control_value( self, state, action, target ):
        control = np.hstack( (np.array(state), np.array(action)) )
        control_vector = self.control_feature_extractor.transform( [control] )
        self.control_value_approximator.partial_fit( X=control_vector, y=[target] )

    def choose_random_action(self):
        random_action_id = np.random.randint( 0, len( self._actions ) )
        random_action = self._id_to_action[random_action_id]
        return random_action

    def choose_greedy_action(self, state):
        action_values = self.get_state_action_values(state).flatten()
        best_action_id = np.argmax( action_values )
        best_action = self._id_to_action[best_action_id]
        return best_action

    # Envrioment Interaction Methods
    def initialize_envrioment(self):
        self.envrioment.initialize()
    
    def state(self):
        return self.envrioment.state()

    def reward(self, action):
        return self.envrioment.reward(action)

    def is_terminal(self):
        return self.envrioment.is_terminal()

    


    
    