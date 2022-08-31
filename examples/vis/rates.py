import sys
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
    print("**** in colab ****")
    if "/content/pyCFTrackers" not in sys.path:
        print("**** path not set ****")
        sys.path.insert(0, "/content/pyCFTrackers")
        print(sys.path)

import numpy as np
import os

path = '/content/pyTrackers/results'
dir_list = os.listdir(path)

dt_normal = []
dt_viot = []
dt_prob = []
dt_rand = []

for f in dir_list:
    if not '.txt' in f:
        continue

    data = np.loadtxt('/content/pyTrackers/results/{}'.format(f), delimiter=',')

    if 'viot' in f:
        dt_viot.append( np.mean(data[1:,0]-data[:-1,0]) )

    elif 'prob' in f:
        dt_prob.append( np.mean(data[1:,0]-data[:-1,0]) )

    elif 'rand' in f:
        dt_rand.append( np.mean(data[1:,0]-data[:-1,0]) )

    else:
        dt_normal.append( np.mean(data[1:,0]-data[:-1,0]) )

    
print('dimp_rate: ', np.mean(dt_normal))
print('dimp_viot_rate: ', np.mean(dt_viot))
print('dimp_prob_rate: ', np.mean(dt_prob))
print('dimp_rand_rate: ', np.mean(dt_rand))