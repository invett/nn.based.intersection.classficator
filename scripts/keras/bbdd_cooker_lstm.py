import json
import matplotlib

# apt-get install ffmpeg libsm6 libxext6  -y


# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import os
import functions
import pickle
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed

# json que ivan tiene para cada dataset, una linea por cada una de las imgs.
{"delta_seconds": 0.00956112239509821, "frame": 2412, "platform_timestamp": 42899.437407493, "velocity": 0.0,
 "player_id": 259, "player_type": "vehicle.tesla.model3",
 "attributes": {"number_of_wheels": "4", "sticky_control": "true", "object_type": "", "color": "180,180,180",
                "role_name": "hero"},
 "weather_type": "WeatherParameters(cloudiness=30.000000, cloudiness=30.000000, precipitation=40.000000, precipitation_deposits=40.000000, wind_intensity=30.000000, sun_azimuth_angle=250.000000, sun_altitude_angle=20.000000, fog_density=15.000000, fog_distance=50.000000, fog_falloff=0.900000, wetness=80.000000)"}

# player_id : identificador de la secuencia/episodio da carla.

cont = 0

episodes = []  # diccionario con una key que es la lista de los frames
velocity_episode = []
elapsed_seconds = []
x = []
y = []
player_id = []
player_type = []
frames = []
frames_len = []

# IVAN
if False:
    line_p = open(sys.argv[1], 'r')
    line = line_p.readline()
    data = eval(line)

    player_id_aux = data['player_id']

    while len(line) != 0:
        data = eval(line)
        velocity_episode.append(data['velocity'])
        elapsed_seconds.append(data['elapsed_seconds'])
        x.append(data['x'])
        y.append(data['y'])
        frames.append(data['frame'])

        if player_id_aux != data['player_id']:
            player_id.append(data['player_id'])
            player_type.append(data['player_type'])
            episodes.append({'id': player_id[-1], 'velocity': velocity_episode[:-2], 'player_type': player_type[-1],
                             'elapsed_seconds': np.asarray(elapsed_seconds[:-2]) - elapsed_seconds[0], 'x': x[:-2],
                             'y': y[:-2], 'frames': frames[1:-2]})
            velocity_episode = []
            elapsed_seconds = []
            x = []
            y = []
            frames = []
            player_id_aux = data['player_id']
            frames_len.append(len(episodes[-1]['frames']))

        cont = cont + 1  # N of samples counter
        line = line_p.readline()

    print('Number of Episodes: ', len(player_id))
    print('Min num frames: ', min(frames_len))

    num_episodes = len(player_id)

    train_index, valid_index, test_index = np.split(player_id, [int(0.6 * num_episodes), int(0.8 * num_episodes)])
    target_validation = []
    print('len train_index: ', len(train_index))
    print('train_index: ', train_index)
    print('valid_index: ', valid_index)
    print('test_index: ', test_index)

# AUGUSTO
if True:

    with open('ivan_kitti360_warped_train.pickle', 'rb') as handle:
        episodes_train = pickle.load(handle)
    with open('ivan_kitti360_warped_valid.pickle', 'rb') as handle:
        episodes_valid = pickle.load(handle)

    # data to populate
    train_index = [i for i in episodes_train]
    valid_index = [i for i in episodes_valid]
    test_index = [] #[i for i in episodes_test]

show_images_flag = True
if show_images_flag == True:

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #  tf.config.experimental.set_memory_growth(gpu, True)
    # tf.config.experimental.per_process_gpu_memory_fraction = 0.9

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    # model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    # include top false, saca la capa fullyconnected, se queda con la 7x7x512
    model_vgg16 = VGG16(weights='imagenet', include_top=False)

    # define model
    model_lstm = Sequential()
    # Add a LSTM layer with 50 internal units.
    model_lstm.add(GRU(50, activation='relu'))
    # Add a Dense layer with 1 units.
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mse')

    input_data_training, target_training = functions.tensor_evaluation(train_index, episodes_train, model_vgg16)
    input_data_validation, target_validation = functions.tensor_evaluation(valid_index, episodes_valid, model_vgg16)

    np.save('input_data_training.npy', input_data_training)
    np.save('input_data_validation.npy', input_data_validation)
    np.save('target_training.npy', target_training)
    np.save('target_validation.npy', target_validation)

    print('Target Training: ', target_training)
    print('Input shape: ', input_data_training.shape)
    print('Target  Training Size: ', target_training.shape)
    print('Training ...')
    output_model = model_lstm.fit(input_data_training, target_training,
                                  validation_data=(input_data_validation, target_validation), epochs=150, verbose=2,
                                  batch_size=3)

    # Save the weights
    model_lstm.save('./my_model_speed_variable_new')

    functions.show_graphical_results(output_model)

    print('Testing')
    input_data_test, target_test = functions.tensor_evaluation(test_index, episodes, model_vgg16)

    np.save('input_data_test.npy', input_data_test)
    np.save('target_test.npy', target_test)

    for episode_elem, target_episode in zip(input_data_test, target_test):
        episode_elem = np.reshape(episode_elem, (1, episode_elem.shape[0], episode_elem.shape[1]))
        print(episode_elem.shape)
        yhat = model_lstm.predict(episode_elem, verbose=1)
        print([yhat, target_episode])
