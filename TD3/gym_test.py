import gym
from gym.utils.play import play

env = gym.make('Pong-v4', render_mode='rgb_array')
env.reset()
play(env, zoom=3, fps=12)

env.close()
