from collections import defaultdict

import numpy as np
import gym
import matplotlib.pyplot as plt


# random argmax function to break ties randomly
def rand_argmax(vector):
    return np.random.choice(np.flatnonzero(np.isclose(vector, vector.max())))


# Function to choose the next action - epsilon-greedy
def choose_action(state):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = rand_argmax(Q[state, :])
    return action


# Function to learn the Q-value
def update(state, state2, reward, action, action2, done):
    if done:
        Q[state, action] = Q[state, action] + alpha * (reward - Q[state, action])
    else:
        predict = Q[state, action]
        target = reward + gamma * Q[state2, action2]
        Q[state, action] = Q[state, action] + alpha * (target - predict)


# Building the environment
env = gym.make('FrozenLake-v0')

# Defining the learning parameters
total_episodes = 1000000
max_steps = 100  # max step in each episode
alpha = 0.40  # learning rate .40
gamma = 0.99  # decay factor .99
epsilon = 0.05  # for exploration .05
test_episodes = 1000

# Initializing
Q = np.zeros((env.observation_space.n, env.action_space.n))
# Q = dict(lambda: np.zeros(env.action_space.n))  # if num states unknown initially
total_reward = 0
test_reward = 0
ep_reward = np.zeros(total_episodes)
ep_steps = np.zeros(total_episodes)
cum_reward = np.zeros(total_episodes)
ep_reward_test = np.zeros(test_episodes)
ep_steps_test = np.zeros(test_episodes)
cum_reward_test = np.zeros(test_episodes)

# Starting the SARSA learning - train
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
            ep_reward[episode] = reward
            ep_steps[episode] = t
            cum_reward[episode] = total_reward
            env.render()
            break

# Evaluating the performance
print(" --- Training --- ")
print("total eps : ", total_episodes)
print("Performance : ", total_reward / total_episodes)

# Visualizing the Q-matrix
print(Q)

# Q.dump("Q_SARSA.dat")

# plot learning
x = range(total_episodes)
plt.plot(x, cum_reward)
plt.ylabel('cumulative ep reward')
plt.show()


# Testing
for episode in range(test_episodes):
    t = 0
    state1 = env.reset()
    action1 = choose_action(state1)

    while t < max_steps:
        # Choosing the action
        action1 = choose_action(state1)

        # Getting the next state
        state2, reward, done, info = env.step(action1)

        state1 = state2

        # Updating the respective values
        t += 1
        test_reward += reward

        # If at the end of learning process
        if done:
            ep_reward_test[episode] = reward
            ep_steps_test[episode] = t
            cum_reward_test[episode] = test_reward
            env.render()
            break

# Evaluating the performance
print(" --- Testing --- ")
print("total eps : ", test_episodes)
print("Performance : ", test_reward / test_episodes)

# plot test
x = range(test_episodes)

plt.subplot(3, 1, 1)
plt.plot(x, ep_reward_test, '-', lw=2)
plt.ylabel('ep reward')

plt.subplot(3, 1, 2)
plt.plot(x, cum_reward_test, '-', lw=2)
plt.ylabel('cumulative ep reward')

plt.subplot(3, 1, 3)
plt.plot(x, ep_steps_test, '-', lw=2)
plt.xlabel('episodes')
plt.ylabel('steps per episode')

plt.show()
