import os
import sys
import unittest

from RLearning.monte_carlo import MonteCarlo
from RLearning.envrioments import RandomDiscreteWalk

sys.path.append( os.path.join(os.path.dirname(__file__), '..') )

class TestMC(unittest.TestCase):
    def test_mc_tabular_integration( self ):
        mc_method = MonteCarlo( episodes=100 )
        envrioment = RandomDiscreteWalk()

        mc_method.fit( envrioment )
