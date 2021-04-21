import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2

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
      episodes.append({'id': player_id[-1], 'velocity': velocity_episode[:-2], 'player_type': player_type[-1], 'elapsed_seconds': np.asarray(elapsed_seconds[:-2]) - elapsed_seconds[0], 'x': x[:-2], 'y': y[:-2], 'frames': frames[1:-2]})
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

train_index, valid_index, test_index = np.split(player_id, [int(0.6*num_episodes), int(0.8*num_episodes)])
print ('train_index: ', train_index)
print ('valid_index: ', valid_index)
print ('test_index: ', test_index)

show_images_flag = False
if show_images_flag == True:
  cv2.namedWindow('Input')

  for index in train_index:
    episode = list(filter(lambda episode: episode['id'] == index, episodes))[0]
    print (episode['id'])
    print (len(episode['frames']))
    for frame in episode['frames']:
      name = '_out/%015d.png' % frame
      print (name)
      image = cv2.imread(name)
      cv2.imshow('Input', image)
      cv2.waitKey(0)


fig2 = plt.figure(figsize=(14.0, 8.0))
for elem in episodes:
    plt.plot(elem['elapsed_seconds'], elem['velocity'], '.-', label = 'Episode: ' + str(elem['id']) + ' Type: ' + elem['player_type'])
    plt.xlabel('Seconds')
    plt.ylabel('Velocity')
    plt.title('Episode velocity')
    plt.legend()

fig3 = plt.figure(figsize=(14.0, 8.0))
for elem in episodes:
    plt.plot(elem['x'], elem['y'], '.-', label = 'Episode: ' + str(elem['id']) + ' Type: ' + elem['player_type'])
    plt.xlabel('m')
    plt.ylabel('m')
    plt.title('Episode trajectory')
    plt.legend()

plt.show()
