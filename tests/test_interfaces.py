import os
import sys
import unittest

sys.path.append( os.path.join(os.path.dirname(__file__), '..') )

from RLearning.interfaces import TabularInterface, ApproximatedInterface
from RLearning.environment import RandomDiscreteWalk, Random1000StateWalk

from sklearn.linear_model import SGDRegressor
from RLearning.feature_extraction import TileCoding

import numpy as np

class TestTabularInterface(unittest.TestCase):

    def test_fit(self):
        tabular_interface = TabularInterface()
        environment = RandomDiscreteWalk()

        tabular_interface.fit( environment )

    def test_state(self):
        tabular_interface = TabularInterface()
        environment = RandomDiscreteWalk()

        tabular_interface.fit( environment )
        tabular_interface.initialize_environment()
        tabular_interface.state()

    def test_reward(self):
        tabular_interface = TabularInterface()
        environment = RandomDiscreteWalk()

        tabular_interface.fit( environment )
        tabular_interface.initialize_environment()

        tabular_interface.reward( 0 )

class TestApproximatedInterface(unittest.TestCase):
    
    def test_fit_control(self):
        sgd_reg =  SGDRegressor()
        tc_ext = TileCoding( n_bins=[100, 1], limits=[ [0, 1000+1], [0,0] ], tile_shift=[0,0] )
        app_interface = ApproximatedInterface( control_feature_extractor=tc_ext,
                                               control_value_approximator=sgd_reg
                                             )
        environment = Random1000StateWalk()

        app_interface.fit( environment )
        app_interface.initialize_environment()

    def test_state(self):
        sgd_reg =  SGDRegressor()
        tc_ext = TileCoding( n_bins=[100, 1], limits=[ [0, 1000+1], [0,0] ], tile_shift=[0,0] )
        app_interface = ApproximatedInterface( control_feature_extractor=tc_ext,
                                               control_value_approximator=sgd_reg
                                             )
        environment = Random1000StateWalk()

        app_interface.fit( environment )
        app_interface.initialize_environment()
        app_interface.state()

    def test_reward(self):
        sgd_reg =  SGDRegressor()
        tc_ext = TileCoding( n_bins=[100, 1], limits=[ [0, 1000+1], [0,0] ], tile_shift=[0,0] )
        app_interface = ApproximatedInterface( control_feature_extractor=tc_ext,
                                               control_value_approximator=sgd_reg
                                             )
        environment = Random1000StateWalk()

        app_interface.fit( environment )
        app_interface.initialize_environment()
        app_interface.reward( 0 )
    