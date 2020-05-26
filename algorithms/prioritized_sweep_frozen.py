import numpy as np
import gym
import matplotlib.pyplot as plt
from queue import PriorityQueue

# Building the environment
env = gym.make('FrozenLake-v0')

# Defining the parameters
total_episodes = 10000
max_steps = 10  # max step in each episode
alpha = 0.4  # learning rate .85
epsilon = 0.05  # for exploration .05
theta = 0
# test params
test_episodes = 100
ep_reward_test = np.zeros(test_episodes)
ep_steps_test = np.zeros(test_episodes)
cum_reward_test = np.zeros(test_episodes)

queue = PriorityQueue()
predecessors = {}  # nxtState -> list[(curState, Action)...]

# Initializing the Q-matrix
model = {}
Q_values = {}
for i in range(env.observation_space.n):
    Q_values[i] = {}
    for a in range(env.action_space.n):
        Q_values[i][a] = 0
# Initializing the reward
total_reward = 0


# Function to choose the next action - epsilon-greedy
def choose_action(state):
    action = ""
    mx_nxt_reward = -999
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        # if all actions have same value, then select randomly
        if len(set(Q_values[state].values())) == 1:
            action = env.action_space.sample()
        else:
            for a in range(env.action_space.n):
                nxt_reward = Q_values[state][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
    return action


# Starting the prioritized sweeping learning
for episode in range(total_episodes):
    print("ep : ", episode)
    print("tot reward : ", total_reward)
    env.render()

    state_actions = []
    done = False
    state1 = env.reset()

    while not done:
        action1 = choose_action(state1)
        state_actions.append((state1, action1))

        # Getting the next state
        state2, reward, done, info = env.step(action1)

        # update priority queue
        tmp_diff = reward + np.max(list(Q_values[state2].values())) - Q_values[state1][action1]
        if tmp_diff > theta:
            queue.put((-tmp_diff, (state1, action1)))  # -diff -> (state, action) pop the smallest

        # update model & predecessors
        if state1 not in model.keys():
            model[state1] = {}
        model[state1][action1] = (reward, state2)
        if state2 not in predecessors.keys():
            predecessors[state2] = [(state1, action1)]
        else:
            predecessors[state2].append((state1, action1))
        state1 = state2
        total_reward += reward

        # planning - loop n times to randomly update Q-value
        for _ in range(max_steps):
            if queue.empty():
                break
            _state, _action = queue.get()[1]
            _reward, _nxtState = model[_state][_action]
            Q_values[_state][_action] += alpha * (_reward + np.max(list(Q_values[_nxtState].values()))
                                                  - Q_values[_state][_action])

            # loop for all state, action predicted lead to _state
            if _state not in predecessors.keys():
                continue
            pre_state_action_list = predecessors[_state]

            for (pre_state, pre_action) in pre_state_action_list:
                pre_reward, _ = model[pre_state][pre_action]
                pre_tmp_diff = pre_reward + np.max(list(Q_values[_state].values())) - Q_values[pre_state][pre_action]
                if pre_tmp_diff > theta:
                    queue.put((-pre_tmp_diff, (pre_state, pre_action)))


# Evaluating the performance
print("total eps : ", total_episodes)
print("Performance : ", total_reward / total_episodes)

# Visualizing the Q-matrix
print(Q_values)


# Testing
test_reward = 0
for episode in range(test_episodes):
    done = False
    state1 = env.reset()
    action1 = choose_action(state1)

    while not done:
        # Choosing the action
        action1 = choose_action(state1)

        # Getting the next state
        state2, reward, done, info = env.step(action1)

        # Updating the respective values
        test_reward += reward

        # If at the end of learning process
        if done:
            ep_reward_test[episode] = reward
            cum_reward_test[episode] = test_reward
            # env.render()
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
