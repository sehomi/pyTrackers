import sys
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
    print("**** in colab ****")
    if "/content/pyTrackers" not in sys.path:
        print("**** path not set ****")
        sys.path.insert(0, "/content/pyTrackers")
        print(sys.path)

import matplotlib.pyplot as plt
import numpy as np
import os
from lib.utils import get_ground_truthes_viot


path = "../../dataset/VIOT/"
dir_list = os.listdir(path)


########## plotting motion speeds #################### 
    
dr = "park_mavic_4"
content = np.genfromtxt(path+dr+"/camera_states.txt", delimiter=',')

xyz = content[1:, 1:4]
rpy = content[1:, 4:7]*180/3.1415
ts = content[1:, 0] - content[0, 0]

dts = content[1:, 0] - content[0:-1, 0]
rates = content[1:, 4:7] - content[0:-1, 4:7]
rates[:,0] = rates[:,0] / dts
rates[:,1] = rates[:,1] / dts
rates[:,2] = rates[:,2] / dts
rates = rates*180/3.1415

plt.rcParams["figure.figsize"] = (7,3)
fig, ax = plt.subplots()

ax.plot(ts, rpy[:,0], label="Roll")
ax.plot(ts, rpy[:,1], label="Pitch")
ax.plot(ts, rpy[:,2], label="Yaw")
ax.set_ylabel('Camera Angles (deg)')
ax.set_xlabel('Time (sec)')
ax.set_xlim([np.min(ts),np.max(ts)])
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig('camera_history.pdf', format="pdf")