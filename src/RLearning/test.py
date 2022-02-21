from monte_carlo import MonteCarlo
from temporal_difference import SARSA, QLearning, ExpectedSARSA
from envrioments import RandomDiscreteWalk

rdw_envrioment = RandomDiscreteWalk()

mc_learner = MonteCarlo(rdw_envrioment)
sarsa_learner = SARSA(rdw_envrioment)
q_learner = QLearning(rdw_envrioment)
e_sarsa_learner = ExpectedSARSA(rdw_envrioment)

print("Fitting MC")
mc_learner.fit(episodes=10000)
print( mc_learner.state_value )

print("Fitting SARSA")
sarsa_learner.fit(episodes=10000)
print( sarsa_learner.state_value )

print("Fitting QLearning")
q_learner.fit(episodes=10000)
print( q_learner.state_value )

print("Fitting Expected SARSA")
e_sarsa_learner.fit(episodes=10000)
print( e_sarsa_learner.state_value )
