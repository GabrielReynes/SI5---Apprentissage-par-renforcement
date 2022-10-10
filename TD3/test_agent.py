from train_agent import model_save_path, feature_range
from gym_record import crop_image, image_args
from keras import models
import numpy as np
import gym

max_step = 1000

if __name__ == '__main__':
    model = models.load_model(model_save_path)
    model.summary()

    env = gym.make('Pong-v4', render_mode='human')
    state, info = env.reset()

    for _ in range(max_step):
        cropped_state = crop_image(state, **image_args) / feature_range

        model_res = model.predict(np.array([cropped_state]), verbose=0)
        action_index = model_res.argmax()
        action = action_index + (action_index > 0)
        print(action)

        state, reward, done, truncated, info = env.step(action)



