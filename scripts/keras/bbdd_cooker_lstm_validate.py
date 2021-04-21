import json
import matplotlib
from matplotlib.cbook import get_sample_data
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
from scipy.stats import norm


import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


{"delta_seconds": 0.00956112239509821, "frame": 2412, "platform_timestamp": 42899.437407493, "velocity": 0.0, "player_id": 259, "player_type": "vehicle.tesla.model3", "attributes": {"number_of_wheels": "4", "sticky_control": "true", "object_type": "", "color": "180,180,180", "role_name": "hero"}, "weather_type": "WeatherParameters(cloudiness=30.000000, cloudiness=30.000000, precipitation=40.000000, precipitation_deposits=40.000000, wind_intensity=30.000000, sun_azimuth_angle=250.000000, sun_altitude_angle=20.000000, fog_density=15.000000, fog_distance=50.000000, fog_falloff=0.900000, wetness=80.000000)"}

cont = 0

episodes = []
velocity_episode = []
elapsed_seconds = []
x = []
y = []
player_id = []
player_type = []
frames = []



line_p = open(sys.argv[1],'r')
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
      episodes.append({'id': player_id[-1], 'velocity': velocity_episode[:-2], 'player_type': player_type[-1], 'elapsed_seconds': np.asarray(elapsed_seconds[:-2]) - elapsed_seconds[0], 'x': x[:-2], 'y': y[:-2], 'frames': frames[1:-2], 'weather': data['weather_type'][18:-2]})
      velocity_episode = []
      elapsed_seconds = []
      x = []
      y = []
      frames = []
      player_id_aux = data['player_id']

    cont = cont + 1 #N of samples counter
    line = line_p.readline()

print ('Number of Episodes: ', len(player_id))

num_episodes = len(player_id)

train_index, valid_index, test_index = np.split(player_id, [int(0.5*num_episodes), int(0.75*num_episodes)])
target_train = []
target_validation = []
print ('train_index: ', train_index)
print ('valid_index: ', valid_index)
print ('test_index: ', test_index)


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

model_vgg16 = VGG16(weights='imagenet', include_top=False)
model_lstm = tf.keras.models.load_model('./my_model_speed_variable')

print ('validation')
errores_episodes = []
velocity_estimated_episodes = []
velocity_target_episodes = []
figure3_data = []

count_aux = 0
for index in valid_index:
  count_aux += 1
  #if count_aux == 40:
  #    break
  figure1_data = []
  episode = list(filter(lambda episode: episode['id'] == index, episodes))[0]
  print ('---------')
  print ([episode['id'], episode['player_type'], episode['weather']])
  print ('Frames in episode: ', len(episode['frames']))
  target_validation.append(episode['velocity'][0])
  first_time_flag = True
  for frame in episode['frames']:
    name = '/media/ivan/8dfd42d1-386d-425d-9205-6a3d23d4ae14/cinemometro_CNN/_out/%015d.png' % frame

    img = image.load_img(name, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features_frame = model_vgg16.predict(x)

    if first_time_flag == True:
       features_episode = features_frame
       first_time_flag = False
    else:
       features_episode = np.concatenate((features_episode, features_frame), axis=0)
      
  input_data_validation = np.reshape(features_episode, (features_episode.shape[0], 7, 7*512))

  yhat = model_lstm.predict(input_data_validation, verbose=0)
  velocity_estimated_episodes.append(np.mean(yhat[int(len(yhat)/2):-2]))
  velocity_target_episodes.append(target_validation[-1])
  errores_episodes.append(abs(np.mean(yhat[int(len(yhat)/2):-2]) - target_validation[-1]))

 #values_list = []
 #for elem in episode['weather'].split(', '):
 #   elem.split('='))
 #figure3_data.append({'target': velocity_target_episodes[-1], 'estimated': velocity_estimated_episodes[-1], 'type': episode['player_type'], 'weather': episode['weather']})

  std = np.std(yhat) 
  mean = np.mean(yhat)    
  yhat_sorted = np.asarray(sorted(yhat))
  fit = norm.pdf(yhat_sorted,mean,std)

  fit_aux = np.concatenate((fit, yhat_sorted), axis=1)

  count = 0
  for frame, measure_speed in zip(episode['frames'], yhat):
    if count % 10 == 0:
      index = np.where(fit_aux[:,1] == measure_speed)[0]
      name = '/media/ivan/8dfd42d1-386d-425d-9205-6a3d23d4ae14/cinemometro_CNN/_out/%015d.png' % frame
      figure1_data.append({'name': name, 'measure_speed': measure_speed[0], 'frame': count, 'fit_value': fit_aux[index,0][0]})
    count += 1
    

  left_margin = 0.04
  botton_margin = 0.06

  fig2 = plt.figure(figsize=(14.0, 8.0))
  plt.subplots_adjust(left=left_margin, bottom=botton_margin, right=0.99, top=0.95, wspace=0, hspace=0)
  plt.plot(np.asarray(yhat), '.-', label='LSTM estimation')
  plt.plot(target_validation[-1]*np.ones(len(yhat)), '.-', label='Target')
  plt.xlabel('Frame')
  plt.ylabel('Velocity (m/s)')
  plt.title('Episode ' + str(episode['id']) + ' Type: ' + str(episode['player_type']))
  plt.xlim([0, count])
  plt.ylim([min(yhat)[0], max(yhat)[0]])
  plt.legend()

  for data in figure1_data:
    im = plt.imread(get_sample_data(data['name']))
    x = data['frame']
    y = data['measure_speed']
    fig2.axes[0].annotate('', xy=(x, y), xytext=(x+1, y+1), arrowprops=dict(facecolor='black', arrowstyle='->'))
    xdisplay, ydisplay = fig2.axes[0].transData.transform((x+1, y+1))
    xfig, yfig = fig2.transFigure.inverted().transform((xdisplay, ydisplay))
    newax = fig2.add_axes([xfig, yfig, 0.05, 0.05], anchor='NE', zorder=1)
    newax.imshow(im)
    newax.axis('off')


  fig1 = plt.figure(figsize=(14.0, 8.0))
  plt.subplots_adjust(left=left_margin, bottom=botton_margin, right=0.99, top=0.95, wspace=0, hspace=0)
  plt.plot(yhat_sorted,fit,'-o')
  n, bins, a = plt.hist(yhat_sorted, bins=10, density=True)
  plt.plot([target_validation[-1],target_validation[-1]], [0, max(n)],'-', label='Target')
  plt.xlabel('Velocity (m/s)')
  plt.title('Episode ' + str(episode['id']) + ' Type: ' + str(episode['player_type']))
  plt.xlim([min(yhat)[0], max(yhat)[0]])
  plt.ylim([0, 1.3*max(n)])
  plt.legend()

  figure2_data = sorted(figure1_data, key=lambda k: k['measure_speed'])
  for data in figure2_data:
    im = plt.imread(get_sample_data(data['name']))
    x = data['measure_speed']
    y = data['fit_value'] 
    fig1.axes[0].annotate('', xy=(x, y), xytext=(x, y+0.1*max(n)), arrowprops=dict(facecolor='black', arrowstyle='->'))
    xdisplay, ydisplay = fig1.axes[0].transData.transform((x, y+0.1*max(n)))
    xfig, yfig = fig1.transFigure.inverted().transform((xdisplay, ydisplay))
    newax = fig1.add_axes([xfig, yfig, 0.05, 0.05], anchor='NE', zorder=1)
    newax.imshow(im)
    newax.axis('off')

  # Save files in pdf and eps format
  fig1.savefig('figures/Episode_' + str(episode['id']) + 'temporalSpeed.eps', dpi=150)
  fig2.savefig('figures/Episode_' + str(episode['id']) + 'statisticalSpeed.eps', dpi=150)


if 1:
  labels = []
  target_means = []
  estimated_means = []
  weathers = []
  for elem in figure3_data:
    labels.append(str(elem['type']))
    target_means.append(elem['target'])
    estimated_means.append(elem['estimated'])
    weathers.append(elem['weather'])

  print (labels)
  x = np.arange(len(labels))  # the label locations
  width = 0.10  # the width of the bars

  fig3 = plt.figure(figsize=(14.0, 8.0))
  plt.subplots_adjust(left=left_margin, bottom=botton_margin, right=0.99, top=0.95, wspace=0, hspace=0)
  plt.title('Velocity in episodes')
  rects1 = fig3.axes[0].bar(x - width/2, target_means, width, label='Target')
  rects2 = fig3.axes[0].bar(x + width/2, estimated_means, width, label='Estimated')
  
  # Add some text for labels, title and custom x-axis tick labels, etc.
  fig3.axes[0].set_ylabel('Car velocity (m/s)')
  fig3.axes[0].set_xticks(x)
  fig3.axes[0].set_xticklabels(labels, rotation=45, ha='right')
  fig3.axes[0].legend()

  for elem_x, elem_target, elem_weather in zip(x, target_means, weathers):
      plt.text(elem_x, elem_target, elem_weather[:15], size=8, rotation=45.,
           ha="left", va="center",
           bbox=dict(boxstyle="round",
                     ec=(1., 0.5, 0.5),
                     fc=(1., 0.8, 0.8),
                     )
           )
    
  fig3.tight_layout()
  fig3.savefig('figures/SpeedVsType.eps', dpi=150)
  
#  plt.show()
  

print ('Error_medio: ', np.mean(errores_episodes))
print ('N de errores: ', len(errores_episodes))
