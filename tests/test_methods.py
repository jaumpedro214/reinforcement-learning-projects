import os
import sys
import unittest

from RLearning.monte_carlo import MonteCarlo
from RLearning.temporal_difference import SARSA, QLearning, ExpectedSARSA, NStepSarsa

from RLearning.interfaces import ApproximatedInterface
from RLearning.envrioments import RandomDiscreteWalk, Random1000StateWalk

from sklearn.linear_model import SGDRegressor
from RLearning.feature_extraction import TileCoding

sys.path.append( os.path.join(os.path.dirname(__file__), '..') )

class TestMC(unittest.TestCase):
    def test_mc_tabular_integration( self ):
        mc_method = MonteCarlo( episodes=100 )
        envrioment = RandomDiscreteWalk()

        mc_method.fit( envrioment )

    def test_mc_app_integration( self ):
        sgd_reg =  SGDRegressor()
        tc_ext = TileCoding( n_bins=[100, 1], limits=[ [0, 1000+1], [0,0] ], tile_shift=[0,0] )
        app_interface = ApproximatedInterface( control_feature_extractor=tc_ext,
                                               control_value_approximator=sgd_reg
                                             )
        envrioment = Random1000StateWalk()
        mc_method = MonteCarlo( env_interface=app_interface, episodes=100 )
        mc_method.fit( envrioment )


class TestSARSA( unittest.TestCase ):
    def test_sarsa_tabular_integration( self ):
        sarsa = SARSA( episodes=100 )
        envrioment = RandomDiscreteWalk()

        sarsa.fit( envrioment )

    def test_sarsa_app_integration( self ):
        sgd_reg =  SGDRegressor()
        tc_ext = TileCoding( n_bins=[100, 1], limits=[ [0, 1000+1], [0,0] ], tile_shift=[0,0] )
        app_interface = ApproximatedInterface( control_feature_extractor=tc_ext,
                                               control_value_approximator=sgd_reg
                                             )
        envrioment = Random1000StateWalk()
        sarsa = SARSA( env_interface=app_interface, episodes=100 )
        sarsa.fit( envrioment )


class TestQLearning( unittest.TestCase ):
    def test_ql_tabular_integration( self ):
        qlearning = QLearning( episodes=100 )
        envrioment = RandomDiscreteWalk()
        qlearning.fit( envrioment )

    def test_ql_app_integration( self ):
        sgd_reg =  SGDRegressor()
        tc_ext = TileCoding( n_bins=[100, 1], limits=[ [0, 1000+1], [0,0] ], tile_shift=[0,0] )
        app_interface = ApproximatedInterface( control_feature_extractor=tc_ext,
                                               control_value_approximator=sgd_reg
                                             )
        envrioment = Random1000StateWalk()
        qlearning = QLearning( env_interface=app_interface, episodes=100 )
        qlearning.fit( envrioment )


class TestExpectedSARSA( unittest.TestCase ):
    def test_exp_sarsa_tabular_integration( self ):
        sarsa = ExpectedSARSA( episodes=100 )
        envrioment = RandomDiscreteWalk()

        sarsa.fit( envrioment )

    def test_exp_sarsa_app_integration( self ):
        sgd_reg =  SGDRegressor()
        tc_ext = TileCoding( n_bins=[100, 1], limits=[ [0, 1000+1], [0,0] ], tile_shift=[0,0] )
        app_interface = ApproximatedInterface( control_feature_extractor=tc_ext,
                                               control_value_approximator=sgd_reg
                                             )
        envrioment = Random1000StateWalk()
        sarsa = ExpectedSARSA( env_interface=app_interface, episodes=100 )
        sarsa.fit( envrioment )


class TestNStepSarsa( unittest.TestCase ):
    def test_nstep_sarsa_tabular_integration( self ):
        sarsa = NStepSarsa( episodes=100, n_steps=10 )
        envrioment = RandomDiscreteWalk()

        sarsa.fit( envrioment )

    def test_nstep_sarsa_app_integration( self ):
        sgd_reg =  SGDRegressor()
        tc_ext = TileCoding( n_bins=[100, 1], limits=[ [0, 1000+1], [0,0] ], tile_shift=[0,0] )
        app_interface = ApproximatedInterface( control_feature_extractor=tc_ext,
                                               control_value_approximator=sgd_reg
                                             )
        envrioment = Random1000StateWalk()
        sarsa = NStepSarsa( env_interface=app_interface, episodes=100, n_steps=10 )
        sarsa.fit( envrioment )

    def test_nstep_sarsa_tabular_nsteps_1( self ):
        sarsa = NStepSarsa( episodes=100, n_steps=1 )
        envrioment = RandomDiscreteWalk()
        sarsa.fit( envrioment )