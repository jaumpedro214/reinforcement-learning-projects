from abc import ABC, abstractmethod
import numpy as np

class BaseMethod(ABC):
    def __init__(self, episodes=1, discount=1):
        self.episodes = episodes
        self.discount = discount

    def fit(self, interface):
        for episode in range( self.episodes ):
            interface.initialize_envrioment()
            self.simulate()

    @abstractmethod
    def simulate(self):
        """
        Simulate a single episode
        """
        pass

    @abstractmethod
    def action(self, state_id):
        pass