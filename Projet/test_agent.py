from keras import models
import gym

from Projet.reinforcementLearning.reinforcement_agent import DeepQAgent
from train_agent import model_save_path

if __name__ == '__main__':
    model = models.load_model(model_save_path)
    model.summary()

    env = gym.make("ALE/MsPacman-ram-v5", render_mode='human')
    state, info = env.reset()

    agent = DeepQAgent(model, env, 0)

    done, truncated = False, False

    while not (done or truncated):
        action = agent.get_action(state, 0.1)
        state, reward, done, truncated, info = env.step(action)



