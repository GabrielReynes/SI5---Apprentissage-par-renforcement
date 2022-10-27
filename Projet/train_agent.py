import gym

import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Dense

from reinforcementLearning.reinforcement_agent import DeepQAgent

from reinforcementLearning.epsilon_strategies import epsilon_decay, epsilon_fixed, epsilon_linear, epsilon_exp


model_save_path = "agents/epsilon_static"

if __name__ == '__main__':
    env = gym.make("ALE/MsPacman-ram-v5", render_mode='rgb_array')

    input_dim = env.observation_space.shape[0]
    output_size = env.action_space.n

    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_size, activation='linear'))

    model.summary()
    model.compile(loss='mse', optimizer='adam')

    agent = DeepQAgent(model, env, 100, learn_batch_size=10, death_reward=-100, update_rate=1e-2,
                       epsilon_strategy=epsilon_exp(1.0, 0.1, 0.1))

    history = agent.learn(1_000, 0.9, verbose=0, workers=2)

    model.save(model_save_path)

    cumulative_average = history['cumulative_average']
    epsilon_values = history['epsilon_values']
    plt.plot(range(len(cumulative_average)), cumulative_average)
    plt.plot(range(len(epsilon_values)), epsilon_values)
    plt.show()

    env.close()
