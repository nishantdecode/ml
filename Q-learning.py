import numpy as np

class Environment:
    def __init__(self):
        self.num_states = 6  # Number of states
        self.num_actions = 2  # Number of actions
        self.reward_matrix = np.array([
            [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [10, -1]
        ])  # Reward matrix
        self.transition_matrix = np.array([
            [1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [5, 5]
        ])  # Transition matrix

    def get_reward(self, state, action):
        return self.reward_matrix[state][action]

    def get_next_state(self, state, action):
        return self.transition_matrix[state][action]

class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            # Exploration: choose a random action
            return np.random.choice(self.num_actions)
        else:
            # Exploitation: choose the action with the highest Q-value
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        old_q_value = self.q_table[state, action]
        td_target = reward + self.discount_factor * np.max(self.q_table[next_state])
        new_q_value = (1 - self.learning_rate) * old_q_value + self.learning_rate * td_target
        self.q_table[state, action] = new_q_value

def main():
    env = Environment()
    agent = QLearningAgent(env.num_states, env.num_actions)
    num_episodes = 50
    for episode in range(num_episodes):
        state = 0  # Starting state
        total_reward = 0
        while state != 5:  # Terminal state
            action = agent.choose_action(state)
            reward = env.get_reward(state, action)
            next_state = env.get_next_state(state, action)
            agent.update_q_table(state, action, reward, next_state)
            total_reward += reward
            state = next_state
        print("Episode:", episode, "Total Reward:", total_reward)

if __name__ == "__main__":
    main()
