import numpy as np

class MarkovDecisionProcess:
    def __init__(self, states, actions, rewards, transitions, discount_factor):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.transitions = transitions
        self.discount_factor = discount_factor
    
    def value_iteration(self, theta=0.0001):
        # Initialize value function
        V = {s: 0 for s in self.states}
        while True:
            delta = 0
            for s in self.states:
                v = V[s]
                V[s] = max(
                    sum(
                        self.transitions[s][a][s1] * (self.rewards[s][a][s1] + self.discount_factor * V[s1])
                        for s1 in self.states if s1 in self.transitions[s][a]
                    ) for a in self.actions
                )
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        
        # Determine optimal policy
        policy = {}
        for s in self.states:
            policy[s] = max(
                self.actions,
                key=lambda a: sum(
                    self.transitions[s][a][s1] * (self.rewards[s][a][s1] + self.discount_factor * V[s1])
                    for s1 in self.states if s1 in self.transitions[s][a]
                )
            )
        
        return V, policy

# Example usage
if __name__ == "__main__":
    # Define states, actions, rewards, transitions, and discount factor
    states = ['s1', 's2', 's3']
    actions = ['a1', 'a2']
    rewards = {
        's1': {'a1': {'s1': 10, 's2': 5}, 'a2': {'s1': -1, 's3': -50}},
        's2': {'a1': {'s1': 10, 's2': -10}, 'a2': {'s2': 0, 's3': 40}},
        's3': {'a1': {'s2': 50, 's3': 0}, 'a2': {'s3': 0}}
    }
    transitions = {
        's1': {'a1': {'s1': 0.8, 's2': 0.2}, 'a2': {'s1': 0.9, 's3': 0.1}},
        's2': {'a1': {'s1': 0.3, 's2': 0.7}, 'a2': {'s2': 0.5, 's3': 0.5}},
        's3': {'a1': {'s2': 0.6, 's3': 0.4}, 'a2': {'s3': 1}}
    }
    discount_factor = 0.9
    
    # Create Markov Decision Process
    mdp = MarkovDecisionProcess(states, actions, rewards, transitions, discount_factor)
    
    # Perform value iteration
    V, policy = mdp.value_iteration()
    
    print("Optimal Value Function:")
    for state, value in V.items():
        print(f"State: {state}, Value: {value}")
    
    print("\nOptimal Policy:")
    for state, action in policy.items():
        print(f"State: {state}, Optimal Action: {action}")
