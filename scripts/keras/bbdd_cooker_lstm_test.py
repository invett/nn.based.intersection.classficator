import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2


import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed

import functions

plt.rcParams.update({'font.size': 16})


{"delta_seconds": 0.00956112239509821, "frame": 2412, "platform_timestamp": 42899.437407493, "velocity": 0.0, "player_id": 259, "player_type": "vehicle.tesla.model3", "attributes": {"number_of_wheels": "4", "sticky_control": "true", "object_type": "", "color": "180,180,180", "role_name": "hero"}, "weather_type": "WeatherParameters(cloudiness=30.000000, cloudiness=30.000000, precipitation=40.000000, precipitation_deposits=40.000000, wind_intensity=30.000000, sun_azimuth_angle=250.000000, sun_altitude_angle=20.000000, fog_density=15.000000, fog_distance=50.000000, fog_falloff=0.900000, wetness=80.000000)"}

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


cont = 0

episodes = []
velocity_episode = []
elapsed_seconds = []
x = []
y = []
player_id = []
player_type = []
frames = []
frames_len = []



line_p = open(sys.argv[1],'r')
line = line_p.readline()
data = eval(line)

player_id_aux = data['player_id']

weather_parameters_list = []
while len(line) != 0:
    data = eval(line)
    velocity_episode.append(data['velocity'])
    elapsed_seconds.append(data['elapsed_seconds'])
    x.append(data['x'])
    y.append(data['y'])
    frames.append(data['frame'])



    if player_id_aux != data['player_id']:
      weather_string = data['weather_type'][18:].replace(')', ' ')
      for elem in weather_string.split(', '):
        aux = elem.split('=')
        value = float(aux[1])
        weather_parameters_list.append([aux[0], value])
      aux_weather = dict(weather_parameters_list)
      if aux_weather['precipitation'] == 60 and aux_weather['precipitation_deposits'] == 100 and aux_weather['sun_altitude_angle'] == 75:
          weather_label = 'Midday_60_100'
      elif aux_weather['precipitation'] == 60 and aux_weather['precipitation_deposits'] == 100 and aux_weather['sun_altitude_angle'] == 15:
          weather_label = 'Sunset_60_100'
      elif aux_weather['precipitation'] == 30 and aux_weather['precipitation_deposits'] == 50 and aux_weather['sun_altitude_angle'] == 75:
          weather_label = 'Midday_30_50'
      elif aux_weather['precipitation'] == 30 and aux_weather['precipitation_deposits'] == 50 and aux_weather['sun_altitude_angle'] == 15:
          weather_label = 'Sunset_30_50'
      elif aux_weather['precipitation'] == 15 and aux_weather['precipitation_deposits'] == 50 and aux_weather['sun_altitude_angle'] == 75:
          weather_label = 'Midday_15_50'
      elif aux_weather['precipitation'] == 15 and aux_weather['precipitation_deposits'] == 50 and aux_weather['sun_altitude_angle'] == 15:
          weather_label = 'Sunset_15_50'
      elif aux_weather['precipitation'] == 40 and aux_weather['precipitation_deposits'] == 40 and aux_weather['sun_altitude_angle'] == 15:
          weather_label = 'Sunset_15_50'
      elif aux_weather['precipitation'] == 0 and aux_weather['precipitation_deposits'] == 50 and aux_weather['sun_altitude_angle'] == 75:
          weather_label = 'Midday_0_50'
      elif aux_weather['precipitation'] == 0 and aux_weather['precipitation_deposits'] == 50 and aux_weather['sun_altitude_angle'] == 15:
          weather_label = 'Sunset_0_50'
      elif aux_weather['precipitation'] == 0 and aux_weather['precipitation_deposits'] == 0 and aux_weather['sun_altitude_angle'] == 75:
          weather_label = 'Midday_0_0'
      elif aux_weather['precipitation'] == 0 and aux_weather['precipitation_deposits'] == 0 and aux_weather['sun_altitude_angle'] == 15:
          weather_label = 'Sunset_0_0'

        
      player_id.append(data['player_id'])
      player_type.append(data['player_type'])
      episodes.append({'id': player_id[-1], 'velocity': velocity_episode[:-2], 'player_type': player_type[-1], 'elapsed_seconds': np.asarray(elapsed_seconds[:-2]) - elapsed_seconds[0], 'x': x[:-2], 'y': y[:-2], 'frames': frames[1:-2], 'weather': dict(weather_parameters_list), 'weather_label': weather_label})
      velocity_episode = []
      elapsed_seconds = []
      x = []
      y = []
      frames = []
      weather_parameters_list = []
      player_id_aux = data['player_id']
      frames_len.append(len(episodes[-1]['frames']))

    cont = cont + 1 #N of samples counter
    line = line_p.readline()

print ('Number of Episodes: ', len(player_id))
print ('Min num frames: ', min(frames_len))

num_episodes = len(player_id)

train_index, valid_index, test_index = np.split(player_id, [int(0.6*num_episodes), int(0.8*num_episodes)])
target_train = []
target_validation = []
print ('len train_index: ', len(train_index))
print ('train_index: ', train_index)
print ('valid_index: ', valid_index)
print ('test_index: ', test_index)

episodes_test = []
for index in test_index[1:]:
    episode = list(filter(lambda episode: episode['id'] == index, episodes))[0]
    episodes_test.append(episode)

weather_label_list = ['Midday_60_100', 'Sunset_60_100', 'Midday_30_50', 'Sunset_30_50', 'Midday_15_50', 'Sunset_15_50', 'Midday_0_50', 'Sunset_0_50', 'Midday_0_0','Sunset_0_0']


# Save the weights
model_vgg16 = VGG16(weights='imagenet', include_top=False)
model_lstm = tf.keras.models.load_model('./my_model_speed_variable_new')

#input_data_test, target_test = functions.tensor_evaluation(test_index, episodes, model_vgg16)
input_data_test = np.load('input_data_test.npy')
input_data_test_3d = np.load('input_data_test_3d.npy')
target_test = np.load('target_test.npy')

print ('validation')
figure3_data = []
print (len(input_data_test))
print (len(episodes_test))

# -------------------------------------------
# Plot error =f(car_type)
# -------------------------------------------
error_list = []
for episode_elem, episode, test_3d in zip(input_data_test, episodes_test, input_data_test_3d[1:,:]):
  episode_elem = np.reshape(episode_elem, (1, episode_elem.shape[0], episode_elem.shape[1]))
  estimated_velocity = model_lstm.predict(episode_elem, verbose=1)
  error = abs(estimated_velocity[0][0] - episode['velocity'][0])
  error_list.append(error)

  figure3_data.append({'target': episode['velocity'][0], 'estimated': estimated_velocity[0][0], 'error': error, 'error_3d': test_3d[0], 'id': episode['id'], 'id_3d': test_3d[1], 'type': episode['player_type'], 'weather_label': episode['weather_label']})

  if figure3_data[-1]['id'] != figure3_data[-1]['id_3d']:
      print ('Error input data')
      exit()


print ('Test samples: ', len(error_list))
print ('Error medio total: ', np.mean(error_list))
  
car_type = []
target_speed = []
estimated_means = []
weathers = []
errors = []
errors_3d = []
for elem in figure3_data[:int(len(figure3_data)/2)-20]:
  car_type.append(str(elem['type']))
  target_speed.append(elem['target'])
  estimated_means.append(elem['estimated'])
  weathers.append(elem['weather_label'])
  errors.append(elem['error'])
  errors_3d.append(elem['error_3d'])

x = np.arange(len(car_type))  # the label locations
width = 0.10  # the width of the bars

left_margin = 0.04
botton_margin = 0.06

fig3 = plt.figure(figsize=(14.0, 8.0))
plt.subplots_adjust(left=left_margin, bottom=botton_margin, right=0.99, top=0.95, wspace=0, hspace=0)
plt.title('Episode weather condition tested')
rects0 = fig3.axes[0].bar(x - width/2, errors, width, label='GRU model')
#rects1 = fig3.axes[0].bar(x - width/2, target_speed, width, label='Target')
rects2 = fig3.axes[0].bar(x + width/2, errors_3d, width, label='3D model')

# Add some text for labels, title and custom x-axis tick labels, etc.
fig3.axes[0].set_ylabel('Error (m/s)')
fig3.axes[0].set_xticks(x)
fig3.axes[0].set_xticklabels(weathers, rotation=65, ha='right')
fig3.axes[0].legend()

for elem_x, elem_y, elem_y_3d, value in zip(x, errors, errors_3d, target_speed):
    aux_y = elem_y
    if elem_y_3d > elem_y:
        aux_y = elem_y_3d

    plt.text(elem_x, aux_y+0.5, round(value, 1), size=16, rotation=45.,
         ha="left", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
  
fig3.tight_layout()
fig3.savefig('figures/ErrorVsWeatherAll.eps', dpi=150)

x = np.arange(len(car_type))  # the label locations
width = 0.10  # the width of the bars

left_margin = 0.04
botton_margin = 0.06

fig3 = plt.figure(figsize=(14.0, 8.0))
plt.subplots_adjust(left=left_margin, bottom=botton_margin, right=0.99, top=0.95, wspace=0, hspace=0)
plt.title('Episode car type tested')
rects0 = fig3.axes[0].bar(x - width/2, errors, width, label='GRU Model')
#rects1 = fig3.axes[0].bar(x - width/2, target_speed, width, label='Target')
rects2 = fig3.axes[0].bar(x + width/2, errors_3d, width, label='3D Model')

# Add some text for labels, title and custom x-axis tick labels, etc.
fig3.axes[0].set_ylabel('Error (m/s)')
fig3.axes[0].set_xticks(x)
fig3.axes[0].set_xticklabels(car_type, rotation=65, ha='right')
fig3.axes[0].legend()

for elem_x, elem_y, elem_y_3d, value in zip(x, errors, errors_3d, target_speed):
    aux_y = elem_y
    if elem_y_3d > elem_y:
        aux_y = elem_y_3d

    plt.text(elem_x, aux_y+0.5, round(value, 1), size=16, rotation=45.,
         ha="left", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
  
fig3.tight_layout()
fig3.savefig('figures/SpeedVsCarAll.eps', dpi=150)

# -------------------------------------------
# Plot error =f(car_type)
# -------------------------------------------
figure3_data_by_weather = []
for label in weather_label_list:
  figure3_data_by_weather.append(list(filter(lambda episode: episode['weather_label'] == label, figure3_data)))

aux_figure_3 = []
for elem_weather in figure3_data_by_weather:
  car_type = []
  target_speed = []
  estimated_speed = []
  errors = []
  errors_3d = []
  for elem in elem_weather:
    car_type.append(str(elem['type']))
    target_speed.append(elem['target'])
    estimated_speed.append(elem['estimated'])
    errors.append(elem['error'])
    errors_3d.append(elem['error_3d'])
  aux_figure_3.append({'weather_label': elem_weather[0]['weather_label'], 'weather_number': len(target_speed), 'target_speed_mean': np.mean(target_speed), 'error_mean': np.mean(errors), 'error_mean_3d': np.mean(errors_3d)})

weathers = []
errors = []
errors_3d = []
target_speed = []
weather_number = []
for elem in aux_figure_3:
    weathers.append(elem['weather_label'])
    errors.append(elem['error_mean'])
    errors_3d.append(elem['error_mean_3d'])
    target_speed.append(elem['target_speed_mean'])
    weather_number.append(elem['weather_number'])

x = np.arange(len(weathers))  # the label locations
width = 0.10  # the width of the bars

left_margin = 0.04
botton_margin = 0.06

fig3 = plt.figure(figsize=(14.0, 8.0))
plt.subplots_adjust(left=left_margin, bottom=botton_margin, right=0.99, top=0.95, wspace=0, hspace=0)
plt.title('Error velocity by weather conditions')
rects0 = fig3.axes[0].bar(x - width/2, errors, width, label='GRU model')
#rects1 = fig3.axes[0].bar(x - width/2, target_speed, width, label='Target')
rects2 = fig3.axes[0].bar(x + width/2, errors_3d, width, label='3D model')

# Add some text for labels, title and custom x-axis tick labels, etc.
fig3.axes[0].set_ylabel('Error (m/s)')
fig3.axes[0].set_xticks(x)
fig3.axes[0].set_xticklabels(weathers, rotation=65, ha='right')
fig3.axes[0].legend()

for elem_x, elem_y, elem_y_3d, value, value_weather in zip(x, errors, errors_3d, target_speed, weather_number):

    aux_y = elem_y
    if elem_y_3d > elem_y:
        aux_y = elem_y_3d

    plt.text(elem_x + 0.1, aux_y, round(value, 1), size=16, rotation=45.,
         ha="left", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
  
    plt.text(elem_x, aux_y + 0.05, value_weather, size=16, rotation=0.,
         ha="right", va="bottom",
         bbox=dict(boxstyle="round",
                   ec=(0.5, 0.5, 0.5),
                   fc=(0.5, 0.8, 0.8),
                   )
         )
  
fig3.tight_layout()
fig3.savefig('figures/ErrorVsWeather.eps', dpi=150)

# -------------------------------------------
# Plot error =f(car_type)
# -------------------------------------------
car_label_list = ['vehicle.audi.a2',
'vehicle.audi.etron',
'vehicle.audi.tt',
'vehicle.bh.crossbike',
'vehicle.bmw.grandtourer',
'vehicle.bmw.isetta',
'vehicle.carlamotors.carlacola',
#'vehicle.charger2020.charger2020',
#'vehicle.chargercop2020.chargercop2020',
'vehicle.chevrolet.impala',
'vehicle.citroen.c3',
'vehicle.diamondback.century',
'vehicle.dodge_charger.police',
'vehicle.gazelle.omafiets',
'vehicle.harley-davidson.low_rider',
'vehicle.jeep.wrangler_rubicon',
'vehicle.kawasaki.ninja',
'vehicle.lincoln.mkz2017',
#'vehicle.lincoln2020.mkz2020',
'vehicle.mercedes-benz.coupe',
#'vehicle.mercedesccc.mercedesccc',
'vehicle.mini.cooperst',
'vehicle.mustang.mustang',
'vehicle.nissan.micra',
'vehicle.nissan.patrol',
'vehicle.seat.leon',
'vehicle.tesla.cybertruck',
'vehicle.tesla.model3',
'vehicle.toyota.prius',
'vehicle.volkswagen.t2',
'vehicle.yamaha.yzf']

figure3_data_by_car = []
for label in car_label_list:
  figure3_data_by_car.append(list(filter(lambda episode: episode['type'] == label, figure3_data)))

aux_figure_3 = []
for elem_car in figure3_data_by_car:
  target_speed = []
  estimated_speed = []
  errors = []
  errors_3d = []
  for elem in elem_car:
    target_speed.append(elem['target'])
    estimated_speed.append(elem['estimated'])
    errors.append(elem['error'])
    errors_3d.append(elem['error_3d'])
  aux_figure_3.append({'car_type': elem_car[0]['type'], 'car_type_number': len(target_speed), 'target_speed_mean': np.mean(target_speed), 'error_mean': np.mean(errors), 'error_mean_3d': np.mean(errors_3d)})

car_type = []
errors = []
errors_3d = []
target_speed = []
car_type_number = []
for elem in aux_figure_3:
    car_type.append(elem['car_type'])
    errors.append(elem['error_mean'])
    errors_3d.append(elem['error_mean_3d'])
    target_speed.append(elem['target_speed_mean'])
    car_type_number.append(elem['car_type_number'])

x = np.arange(len(car_type))  # the label locations
width = 0.10  # the width of the bars

left_margin = 0.04
botton_margin = 0.06

fig3 = plt.figure(figsize=(14.0, 8.0))
plt.subplots_adjust(left=left_margin, bottom=botton_margin, right=0.99, top=0.95, wspace=0, hspace=0)
plt.title('Error Velocity by car type')
rects0 = fig3.axes[0].bar(x - width/2, errors, width, label='GRU model')
#rects1 = fig3.axes[0].bar(x - width/2, target_speed, width, label='Target')
rects2 = fig3.axes[0].bar(x + width/2, errors_3d, width, label='3D model')

# Add some text for labels, title and custom x-axis tick labels, etc.
fig3.axes[0].set_ylabel('Error (m/s)')
fig3.axes[0].set_xticks(x)
fig3.axes[0].set_xticklabels(car_type, rotation=65, ha='right')
fig3.axes[0].legend()

for elem_x, elem_y, elem_y_3d, value, car_number in zip(x, errors, errors_3d, target_speed, car_type_number):

    aux_y = elem_y
    if elem_y_3d > elem_y:
        aux_y = elem_y_3d

    plt.text(elem_x + 0.1, aux_y, round(value, 1), size=16, rotation=45., label='Target mean velocity',
         ha="left", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )

    plt.text(elem_x, aux_y + 0.3, car_number, size=16, rotation=0.,
         ha="right", va="bottom",
         bbox=dict(boxstyle="round",
                   ec=(0.5, 0.5, 0.5),
                   fc=(0.5, 0.8, 0.8),
                   )
         )
  
fig3.tight_layout()
fig3.savefig('figures/ErrorVsCarType.eps', dpi=150)
