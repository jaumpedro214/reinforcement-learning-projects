from base.base_tabular_methods import BaseTabularMethod
import numpy as np

class SARSA(BaseTabularMethod):
  def __init__(self, *args, alpha=0.1, eps=0.1, **kwargs):
    self._alpha = alpha
    self._eps = eps
    super(SARSA, self).__init__(*args, **kwargs)

  def action(self, state):
    if np.random.uniform(0, 1) < self._eps:
      return np.random.randint(0, self._num_actions)

    return self.policy[ state ]

  def fit(self, episodes=10):
    for episode in range(episodes):
      self.envrioment.initialize()
      self.simulate()

    return self

  def simulate(self):
    
    current_state = self.envrioment.state()
    current_action = self.action(current_state)

    while not self.envrioment.is_terminal():
      reward = self.envrioment.reward(current_action)

      next_state = self.envrioment.state()
      next_action = self.action(next_state)

      self.state_value_update(reward, current_state, next_state)
      self.state_action_value_update(current_state, current_action, reward, next_state, next_action)
      self.policy_improvement(current_state)

      current_state=next_state
      current_action=next_action

  def state_value_update(self, reward, current_state, next_state):
    td_error = reward+self.discount*self.state_value[next_state]-self.state_value[current_state]
    self.state_value[current_state] += self._alpha*td_error

  def state_action_value_update(self, current_state, current_action, reward, next_state, next_action):
    td_error = reward
    td_error += self.discount*self.state_action_value[next_state][next_action]
    td_error -= self.state_action_value[current_state][current_action]

    self.state_action_value[current_state][current_action] += self._alpha*td_error

  def policy_improvement(self, state):
    self.policy[state] = np.argmax( self.state_action_value[state].flatten() )

class QLearning(SARSA):
  def __init__(self, *args, **kwargs):
    super(QLearning, self).__init__(*args, **kwargs)

  def simulate(self):
    current_state = self.envrioment.state()
    while not self.envrioment.is_terminal():
      action = self.action(current_state)
      reward = self.envrioment.reward(action)
      next_state = self.envrioment.state()

      self.state_value_update(reward, current_state, next_state)
      self.state_action_value_update(current_state, action, reward, next_state)
      self.policy_improvement(current_state)

      current_state=next_state

  def state_action_value_update(self, current_state, action, reward, next_state):
    td_error = reward
    td_error += self.discount*np.max( self.state_action_value[next_state][:] )
    td_error -= self.state_action_value[current_state][action]

    self.state_action_value[current_state][action] += self._alpha*td_error

class ExpectedSARSA(SARSA):
  def __init__(self, *args, **kwargs):
    super(ExpectedSARSA, self).__init__(*args, **kwargs)

  def state_action_value_update(self, current_state, current_action, reward, next_state, next_action):
    td_error = reward
    td_error += self.discount*self._expected_state_reward(next_state)
    td_error -= self.state_action_value[current_state][current_action]

    self.state_action_value[current_state][current_action] += self._alpha*td_error

  def _policy_state_action_probability(self, state, action):
    # This class is eps-greedy
    # p -> eps/|actions| for non-optimal actions
    # p -> 1-eps + eps/|actions| for the optimal action
    p = self._eps/self._num_actions
    if self.policy[state] == action:
      p += 1-self._eps
    return p

  def _expected_state_reward(self, state):
    action_probabilities = np.array( [ self._policy_state_action_probability( state, action )
                                       for action in self._actions ]
                                   )
    
    return (self.state_action_value[state, :]*action_probabilities).sum()