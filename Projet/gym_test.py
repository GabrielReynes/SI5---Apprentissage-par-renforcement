import gym
from gym.utils.play import play

env = gym.make("ALE/MsPacman-ram-v5", render_mode='human')
env.reset()

play(env, zoom=3, fps=12)

env.close()
