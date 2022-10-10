import numpy as np
from keras import layers, utils
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from tensorflow_addons.metrics import F1Score
from sklearn.model_selection import train_test_split
from tensorflow import keras

from gym_record import image_args, label_out_file_path, sample_out_file_path

model_save_path = "agents/first-model"
nb_classes = 3
feature_range = 256

f1_metric = F1Score(num_classes=nb_classes, average='weighted')

callbacks_list = [EarlyStopping(monitor='val_f1_score', min_delta=0.0005,
                                patience=20, verbose=1, mode='max',
                                restore_best_weights=True)]

if __name__ == '__main__':
    feature_width = (image_args['x_axis_max'] - image_args['x_axis_min']) // image_args['x_axis_step']
    feature_height = (image_args['y_axis_max'] - image_args['y_axis_min']) // image_args['y_axis_step']

    X_input = open(sample_out_file_path, 'r')
    y_input = open(label_out_file_path, 'r')

    X = np.array(list(map(lambda s: list(map(int, s.split())), X_input))).reshape(-1, feature_height, feature_width) / feature_range
    y = utils.to_categorical(list(map(int, y_input)), nb_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = keras.Sequential()
    model.add(layers.Conv2D(32, 5, input_shape=(feature_height, feature_width, 1), activation='relu'))
    model.add(layers.Conv2D(16, 3, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(nb_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X_train, y_train, epochs=10, validation_split=0.1, verbose=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = keras.Sequential()
    model.add(layers.Conv2D(32, 5, input_shape=(feature_height, feature_width, 1), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(nb_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1_metric])
    model.fit(X_train, y_train, epochs=100, batch_size=20, validation_split=0.33, verbose=1, callbacks=callbacks_list)

    y_pred = model.predict(X_test)

    flat_y_test = np.argmax(y_test, axis=1)
    flat_y_pred = np.argmax(y_pred, axis=1)

    print(f1_score(flat_y_test, flat_y_pred, average='micro'))
    print(multilabel_confusion_matrix(flat_y_test, flat_y_pred))

    model.save(model_save_path)
