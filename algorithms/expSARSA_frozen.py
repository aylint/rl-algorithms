import numpy as np
import gym


# Defining the parameters
total_episodes = 10000
max_steps = 100  # max step in each episode
alpha = 0.30  # learning rate
gamma = 0.99  # decay factor
epsilon = 0.1  # for exploration
# Initializing the Q-matrix
Q = np.zeros((env.observation_space.n, env.action_space.n))
# Initializing the reward
total_reward = 0


# Function to choose the next action - epsilon-greedy
def choose_action(state):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


# Function to learn the Q-value
def update(state, state2, reward, action, action2, done):
    if done:
        Q[state, action] = Q[state, action] + alpha * (reward - Q[state, action])
    else:
        exp_q = np.mean(Q[state2, :])
        Q[state, action] = Q[state, action] + alpha * (reward + gamma*exp_q - Q[state, action])


# Building the environment
env = gym.make('FrozenLake-v0')

# Starting the SARSA learning
for episode in range(total_episodes):
    t = 0
    state1 = env.reset()
    action1 = choose_action(state1)

    while t < max_steps:
        # Visualizing the training
        # env.render()

        # Getting the next state
        state2, reward, done, info = env.step(action1)

        # Choosing the next action
        action2 = choose_action(state2)

        # Learning the Q-value
        update(state1, state2, reward, action1, action2, done)

        state1 = state2
        action1 = action2

        # Updating the respective values
        t += 1
        total_reward += reward

        # If at the end of learning process
        if done:
            env.render()
            break

# Evaluating the performance
print("total eps : ", total_episodes)
print("Performance : ", total_reward / total_episodes)

# Visualizing the Q-matrix
print(Q)
