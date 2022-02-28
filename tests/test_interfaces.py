import os
import sys
import unittest

sys.path.append( os.path.join(os.path.dirname(__file__), '..') )

from RLearning.interfaces import TabularInterface
from RLearning.envrioments import RandomDiscreteWalk
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
