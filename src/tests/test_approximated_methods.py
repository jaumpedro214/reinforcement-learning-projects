import os
import sys
import unittest

sys.path.append( os.path.join(os.path.dirname(__file__), '..') )

from RLearning.feature_extraction import TileCoding
from RLearning.monte_carlo import MonteCarloApproximated
from RLearning.envrioments import Random1000StateWalk, MontainCar

from sklearn.linear_model import SGDRegressor

import numpy as np

class TestAppMonteCarlo(unittest.TestCase):

    def test_control_is_working_random_walk(self):
        envrioment = Random1000StateWalk()
        control_feature_extractor = TileCoding( n_bins=[10, 2], 
                                                limits=[ [1, 1000], [0,1] ],
                                                n_tiles = 1,
                                                tile_shift=[0,0]
                                                ).fit()
        control_value_approximator = SGDRegressor()

        mc_app_learner = MonteCarloApproximated( envrioment,
                                                 control_feature_extractor,
                                                 control_value_approximator
                                               )
        mc_app_learner.fit(episodes=1000)                
