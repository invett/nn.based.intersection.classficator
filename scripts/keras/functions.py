import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import random


def tensor_evaluation(train_index, episodes, model):
    target_train = []
    count = 0
    first_time_input = True
    for index in train_index[1:]:
        print('------------------------------')
        first_time_flag = True
        count += 1
        # episode = list(filter(lambda episode: episode['id'] == index, episodes))[0]
        episode = episodes[index]
        print('Episode ID: ', episode['id'])
        print('Frames: ', len(episode['frames']))
        # target_train.append([episode['velocity'][0]])
        target_train.append(int(episode['gt']))
        for frame in episode['frames']:
            # name = '_out/%015d.png' % frame
            # img = image.load_img(name, target_size=(224, 224))

            img = image.load_img(frame[0], target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            features_frame = model.predict(x)

            if first_time_flag == True:
                features_episode = features_frame
                first_time_flag = False
            else:
                features_episode = np.concatenate((features_episode, features_frame), axis=0)

        timesteps = features_episode.shape[0]
        features = features_episode.shape[1] * features_episode.shape[2] * features_episode.shape[3]
        input_data_episode = np.reshape(features_episode, (1, timesteps, features))     # SEQUENCE x FRAMES x FEATURES

        print('Count: ', count)
        print('Input shape: ', input_data_episode.shape)

        if first_time_input == True:
            #input_data_training = input_data_episode[:, :45, :]
            input_data_training = input_data_episode[:, -6:, :]
            first_time_input = False
        else:
            #input_data_training = np.concatenate((input_data_training, input_data_episode[:, :45, :]), axis=0)
            input_data_training = np.concatenate((input_data_training, input_data_episode[:, -6:, :]), axis=0)

        if count == 3000000:
            break

    target = np.asarray(target_train)

    # shuffle lists https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
    support = list(zip(input_data_training, target))
    random.shuffle(support)
    input_data_training_randomized, target_randomized = zip(*support)

    return input_data_training_randomized, target_randomized


def show_graphical_results(output):
    # summarize history for accuracy
    fig1 = plt.figure(figsize=(14.0, 8.0))
    # summarize history for loss
    plt.plot(output.history['loss'], label='Training')
    plt.plot(output.history['val_loss'], label='Validation')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    fig1.savefig('./lossValues.png', dpi=300)

    fig1 = plt.figure(figsize=(14.0, 8.0))
    # summarize history for loss
    plt.plot(output.history['accuracy'], label='Training')
    plt.plot(output.history['val_accuracy'], label='Validation')
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    fig1.savefig('./AccValues.png', dpi=300)
