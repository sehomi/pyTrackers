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

########## plotting occlusion percentage #################### 
# occ_percentages = []
# names = []
# for dr in dir_list:
#     if '.txt' in dr or '.json' in dr:
#         continue
    
#     names.append(dr)
#     content = np.loadtxt(path+dr+"/occlusion.tag")
#     content = np.array(content).astype(np.int32)
#     occ_percentages.append(float(np.sum(content)) / len(content) * 100)

# occ_percentages = np.array(occ_percentages)
# names = np.array(names)

# args = np.argsort(occ_percentages)
# occ_percentages = occ_percentages[args]
# names = names[args]

# plt.rcParams["figure.figsize"] = (7,3)
# plt.bar(names, occ_percentages,
#         width = 0.4)
# plt.xticks(rotation=45, ha="right")
# plt.ylabel('Occluded Frames (%)')
# plt.tight_layout()
# plt.grid()
# plt.savefig('occlusion_percentage.pdf', format="pdf")

########## plotting motion speeds #################### 

# rate_means = []
# names = []
# for dr in dir_list:
#     if '.txt' in dr or '.json' in dr:
#         continue
    
#     names.append(dr)
#     content = np.genfromtxt(path+dr+"/camera_states.txt", delimiter=',')

#     dts = content[1:, 0] - content[0:-1, 0]
#     rates = content[1:, 4:7] - content[0:-1, 4:7]
#     rates[:,0] = rates[:,0] / dts
#     rates[:,1] = rates[:,1] / dts
#     rates[:,2] = rates[:,2] / dts

#     rates = np.linalg.norm(rates, axis=1)
#     rate_means.append(np.mean(rates)*180/3.1415)

# rate_means = np.array(rate_means)
# names = np.array(names)

# args = np.argsort(rate_means)
# rate_means = rate_means[args]
# names = names[args]

# plt.rcParams["figure.figsize"] = (7,3)
# plt.bar(names, rate_means,
#         width = 0.4)
# plt.xticks(rotation=45, ha="right")
# plt.ylabel('Mean Motion (deg/s)')
# plt.tight_layout()
# plt.grid()
# plt.savefig('motion.pdf', format="pdf")

################## plotting comparison to  ##################

motion = []
lowres = []
names = []
for dr in dir_list:
    if '.txt' in dr or '.json' in dr:
        continue
    
    names.append(dr)
    gts = get_ground_truthes_viot(path+dr)
    gts_displacement = gts[1:,0:2] - gts[0:-1,0:2] 

    bl1 = np.abs(gts_displacement[:,0]) > 0.2*gts[0:-1,2]
    bl2 = np.abs(gts_displacement[:,1]) > 0.2*gts[0:-1,2]
    bl = np.logical_or(bl1, bl2)

    lr = gts[:,2] * gts[:,3] < 2500 

    motion.append(np.sum(bl)/ float(len(bl)))

    lowres.append(np.sum(lr)/ float(len(lr)))

print(np.mean(motion), np.mean(lowres))

xs = np.array([2, 6, 10])

plt.rcParams["figure.figsize"] = (5,3.1)
plt.bar(xs - 1, [39,23,9],
        width = 0.8, label = 'OTB100', color='navy')
plt.bar(xs, [23,29,39],
        width = 0.8, label = 'UAV123', color='maroon')
plt.bar(xs + 1, [np.mean(motion)*100, 38, np.mean(lowres)*100],
        width = 0.8, label = 'VIOT', color='green')
plt.xticks(xs+1, ["Fast Motion", "Occlusion", "Low Resolution"], ha="right")
# plt.xticks(x, my_xticks)
plt.ylabel('percentage (%)')
plt.ylim([0,70])
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('attribs.pdf', format="pdf")