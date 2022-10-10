import gym
import numpy as np
from gym.utils.play import play


def crop_image(origin, x_axis_min, x_axis_max, x_axis_step,
               y_axis_min, y_axis_max, y_axis_step,
               color_chanel):
    return origin[y_axis_min:y_axis_max:y_axis_step, x_axis_min:x_axis_max:x_axis_step, color_chanel]


def create_callback(_sample_out_file, _label_out_file, **kwargs):
    last_img = None

    def my_callback(obs_t, obs_tp1, action, reward, terminated, truncated, info):
        nonlocal last_img

        if type(obs_t) is tuple:
            return

        cropped_img = crop_image(obs_t, **kwargs)

        if last_img is not None:
            diff_img = cropped_img.astype(np.int16) - last_img

            img_min = diff_img.min()
            img_range = diff_img.max() - img_min
            if img_range == 0:
                return
            diff_img = ((diff_img - img_min) / img_range * 255).astype(np.uint8)

            # imageio.imwrite("resources/diff.jpg", diff_img)
            # imageio.imwrite("resources/act.jpg", cropped_img)

            np.savetxt(_sample_out_file, X=diff_img.reshape(1, -1), fmt='%03d')
            _label_out_file.write(f'{max(0, action - 1)}\n')

        last_img = cropped_img.astype(np.int16) * 0.75

    return my_callback


image_args = {
    'x_axis_min': 12,
    'x_axis_max': 148,
    'x_axis_step': 2,
    'y_axis_min': 34,
    'y_axis_max': 194,
    'y_axis_step': 4,
    'color_chanel': 1,
}

sample_out_file_path = 'text_files/X.txt'
label_out_file_path = 'text_files/y.txt'

if __name__ == '__main__':
    env = gym.make('Pong-v4', render_mode='rgb_array')
    env.reset()

    sample_out_file = open('text_files/X.txt', 'a')
    label_out_file = open('text_files/y.txt', 'a')

    callback = create_callback(sample_out_file, label_out_file, **image_args)

    gym.utils.play.play(env, zoom=3, fps=12, callback=callback)

    sample_out_file.close()
    label_out_file.close()
