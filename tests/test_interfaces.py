import os
import sys
import unittest

sys.path.append( os.path.join(os.path.dirname(__file__), '..') )

from RLearning.interfaces import TabularInterface, ApproximatedInterface
from RLearning.envrioments import RandomDiscreteWalk, Random1000StateWalk

from sklearn.linear_model import SGDRegressor
from RLearning.feature_extraction import TileCoding

import numpy as np

class TestTabularInterface(unittest.TestCase):

    def test_fit(self):
        tabular_interface = TabularInterface()
        envrioment = RandomDiscreteWalk()

        tabular_interface.fit( envrioment )

    def test_state(self):
        tabular_interface = TabularInterface()
        envrioment = RandomDiscreteWalk()

        tabular_interface.fit( envrioment )
        tabular_interface.initialize_envrioment()
        tabular_interface.state()

    def test_reward(self):
        tabular_interface = TabularInterface()
        envrioment = RandomDiscreteWalk()

        tabular_interface.fit( envrioment )
        tabular_interface.initialize_envrioment()

        tabular_interface.reward( 0 )

class TestApproximatedInterface(unittest.TestCase):
    
    def test_fit_control(self):
        sgd_reg =  SGDRegressor()
        tc_ext = TileCoding( n_bins=[100, 1], limits=[ [0, 1000+1], [0,0] ], tile_shift=[0,0] )
        app_interface = ApproximatedInterface( control_feature_extractor=tc_ext,
                                               control_value_approximator=sgd_reg
                                             )
        envrioment = Random1000StateWalk()

        app_interface.fit( envrioment )
        app_interface.initialize_envrioment()

    def test_state(self):
        sgd_reg =  SGDRegressor()
        tc_ext = TileCoding( n_bins=[100, 1], limits=[ [0, 1000+1], [0,0] ], tile_shift=[0,0] )
        app_interface = ApproximatedInterface( control_feature_extractor=tc_ext,
                                               control_value_approximator=sgd_reg
                                             )
        envrioment = Random1000StateWalk()

        app_interface.fit( envrioment )
        app_interface.initialize_envrioment()
        app_interface.state()

    def test_reward(self):
        sgd_reg =  SGDRegressor()
        tc_ext = TileCoding( n_bins=[100, 1], limits=[ [0, 1000+1], [0,0] ], tile_shift=[0,0] )
        app_interface = ApproximatedInterface( control_feature_extractor=tc_ext,
                                               control_value_approximator=sgd_reg
                                             )
        envrioment = Random1000StateWalk()

        app_interface.fit( envrioment )
        app_interface.initialize_envrioment()
        app_interface.reward( 0 )
    