import pickle

import gym
import numpy as np

use_serialized = True

env = gym.make('FrozenLake-v1', render_mode='human' if use_serialized else None,
               is_slippery=False, map_name='8x8')

epsilon = 0.1 if use_serialized else 0.9
total_episodes = 100_000
max_steps = 1000
displacement_malus = 0.0

lr_rate = 0.81
gamma = 0.96

Q = np.zeros((env.observation_space.n, env.action_space.n))

if use_serialized:
    with open("frozenLake_qTable.pkl", 'rb') as f:
        Q = pickle.load(f)


def choose_action(_state):
    # action = 0
    if np.random.uniform(0, 1) < epsilon:
        _action = env.action_space.sample()
    else:
        _action = np.argmax(Q[_state, :])
    return _action


def learnQ(_state, _state2, _reward, _action):
    predict = Q[_state, _action]
    target = _reward + gamma * np.max(Q[_state2, :])
    Q[_state, _action] = predict + lr_rate * (target - predict)


def learnSARSA(_state, _state2, _reward, _action, _action2):
    predict = Q[_state, _action]
    target = _reward + gamma * Q[state2, _action2]
    Q[_state, _action] = predict + lr_rate * (target - predict)


# Start
for episode in range(total_episodes):
    state, info = env.reset()
    t = 0

    print(episode)

    while t < max_steps:

        action = choose_action(state)

        state2, reward, done, truncated, info = env.step(action)

        learnQ(state, state2, reward - displacement_malus, action)

        # action2 = choose_action(state2)
        # learnSARSA(state, state2, reward - displacement_malus, action, action2)

        state = state2

        t += 1

        if done or truncated:
            break

        # time.sleep(0.1)

print(Q)

if not use_serialized:
    with open("frozenLake_qTable.pkl", 'wb') as f:
        pickle.dump(Q, f)
