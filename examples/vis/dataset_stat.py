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

path = "../../dataset/VIOT/"
dir_list = os.listdir(path)

########## plotting occlusion percentage #################### 
occ_percentages = []
names = []
for dr in dir_list:
    if '.txt' in dr or '.json' in dr:
        continue
    
    names.append(dr)
    content = np.loadtxt(path+dr+"/occlusion.tag")
    content = np.array(content).astype(np.int32)
    occ_percentages.append(float(np.sum(content)) / len(content) * 100)

occ_percentages = np.array(occ_percentages)
names = np.array(names)

args = np.argsort(occ_percentages)
occ_percentages = occ_percentages[args]
names = names[args]

plt.bar(names, occ_percentages, color ='maroon',
        width = 0.4)
plt.xticks(rotation=45, ha="right")
plt.ylabel('Occluded Frames (%)')
plt.tight_layout()
plt.grid()
plt.savefig('occlusion_percentage.pdf', format="pdf")

########## plotting motion speeds #################### 

