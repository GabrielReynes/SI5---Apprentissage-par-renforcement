import random
import numpy as np
from .agent_memory import AgentMemory
from .epsilon_strategies import epsilon_fixed
from keras.models import clone_model


class DeepQAgent:
    """
    Represents a Game Agent applying a depp Q learning model strategy.
    This agent is defined by an ANN model, a gym library environment, and a memory size.
    You can also set deeper parameters to modify the agent behavior as :
        - The "death_reward" which will be replacing the reward given by the gym environment
            in case of the agent death in game.
        - The "update_rate" which defines the % rate at which the agent will update its target model and epsilon value.
        - The "epsilon_max" and "epsilon_min" values which will be used to define the evolution of the agent epsilon
            value through its learning phase.
    """

    def __init__(self, model, env, memory_size, **kwargs):
        self.policy_model = model
        self.target_model = clone_model(model)
        self.env = env
        self.last_state, info = env.reset()
        self.lives = info['lives']
        self.memory = AgentMemory(memory_size)
        self.learn_batch_size = kwargs.get('learn_batch_size', memory_size)
        self.death_reward = kwargs.get('death_reward', 0)
        self.update_rate = kwargs.get('update_rate', 0.1)
        self.epsilon_max = kwargs.get('epsilon_max', 1.0)
        self.epsilon_min = kwargs.get('epsilon_min', 0.0)
        self.epsilon_strategy = kwargs.get('epsilon_strategy', epsilon_fixed(self.epsilon_max))

    def get_action(self, state, epsilon):
        """
        :param state: The state inside which the agent should pick an action.
        :param epsilon: The epsilon value used to determine at what rate the agent should pick a random action.
        :return: the actions index picked by the agent for a given state.
        """
        if random.random() < epsilon:
            return self.env.action_space.sample()

        model_res = self.policy_model.predict(state.reshape(1, -1), verbose=0)
        action_index = np.argmax(model_res)
        return action_index

    def step(self, epsilon):
        """
        :param epsilon: The epsilon value used to determine at what rate the agent should pick a random action.
        :return: - The reward associated with the new state of the agent,
                 - If the environment is in a done state after the agent step
                 - If the environment is in a truncated state after the agent step.
                 - A dictionary containing further information about the new state of the agent.
        """
        action = self.get_action(self.last_state, epsilon)
        next_state, reward, done, truncated, info = self.env.step(action)
        has_died = info['lives'] < self.lives
        if has_died:
            self.lives = info['lives']
            reward = self.death_reward

        self.update(self.last_state, action, reward, next_state, done)
        self.last_state = next_state
        return reward, done, truncated, info

    def get_q_value(self, state):
        """
        Get a state associated q_values for the agent.

        :param state: The state for which to evaluate the q_value
        :return: The Q_Value associated with the given state.
        """
        self.policy_model.predict(state)

    def update(self, *args):
        """
        Push the given arguments inside the agent's memory as a tuple
        :param args:
        """
        self.memory.push(*args)

    def fit(self, discount, **kwargs):
        """
        Samples a random set of values inside the agent's memory and apply back-propagation
        onto the agent's model to correct its q_values corresponding to the contained states.

        :param discount: The discount factor to be applied to evaluated future rewards
        :param kwargs: Keyword argument to be passed to the keras model methods "predict" and "fit"
        :return: The history object returned by the "predict" method of the keras model.
        """
        memory_sample = self.memory.sample(self.learn_batch_size)
        states = np.array([elm[0] for elm in memory_sample])
        next_states = np.array([elm[3] for elm in memory_sample])
        q_values = self.policy_model.predict(states, **kwargs)
        next_q_values = self.target_model.predict(next_states, **kwargs)
        target_values = np.zeros(q_values.shape)
        for idx in range(len(memory_sample)):
            state, action, reward, next_state, done = memory_sample[idx]
            target_values[idx][action] = reward + discount * (not done) * np.amax(next_q_values[idx])
        return self.policy_model.fit(states, target_values, batch_size=self.learn_batch_size, epochs=1, **kwargs)

    def reset(self):
        """
        Resets the agent last registered state and memory.
        """
        self.last_state = self.env.reset()[0]
        self.memory.clear()

    def learn(self, steps, discount, **kwargs):
        """
        Play a given number of step inside the associated gym library environment.
        Trains the associated model to adapt to the observed rewards during the games.

        :param steps: The number of step to play.
        :param discount: The discount factor to be applied to future estimated reward at training.
        :param kwargs: The keyword arguments to be passed to the karas model "predict" and "fit" methods
        :returns: The cumulative average score of the training session.
        """

        update_step_count = max(1, int(steps * self.update_rate))

        epsilon = self.epsilon_max

        agent_history = {"cumulative_average": [0], "epsilon_values": [epsilon], "loss_values": []}

        for step in range(steps):
            if step % update_step_count == 0:
                self.target_model.set_weights(self.policy_model.get_weights())
                print(step)

            epsilon = self.epsilon_strategy(epsilon, step)
            reward, done, truncated, info = self.step(epsilon)

            new_cumulative_average = (reward + agent_history["cumulative_average"][-1] * step) / (step+1)
            agent_history["cumulative_average"].append(new_cumulative_average)
            agent_history["epsilon_values"].append(epsilon)

            if done or truncated:
                self.reset()
                continue

            history = self.fit(discount, **kwargs)
            agent_history["loss_values"].append(history.history["loss"])

        return agent_history
