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

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from lib.utils_ import get_ground_truthes_viot

min_idx = 110
max_idx = 191

gts=get_ground_truthes_viot('/content/pyTrackers/dataset/VIOT/park_mavic_1')

occ = np.loadtxt('/content/pyTrackers/results/occ_park_mavic_1.txt')[min_idx:max_idx]
kcf_hog = np.loadtxt('/content/pyTrackers/results/kcf_park_mavic_1_viot.txt')
ldes = np.loadtxt('/content/pyTrackers/results/ldes_park_mavic_1_viot.txt')
csrdcf = np.loadtxt('/content/pyTrackers/results/csrdcf_park_mavic_1_viot.txt')
strcf = np.loadtxt('/content/pyTrackers/results/strcf_park_mavic_1_viot.txt')
dimp50 = np.loadtxt('/content/pyTrackers/results/dimp50_park_mavic_1_viot.txt')
kys = np.loadtxt('/content/pyTrackers/results/kys_park_mavic_1_viot.txt')
tomp = np.loadtxt('/content/pyTrackers/results/tomp_park_mavic_1_viot.txt')
prdimp50 = np.loadtxt('/content/pyTrackers/results/prdimp50_park_mavic_1_viot.txt')
mixformer = np.loadtxt('/content/pyTrackers/results/mixformer_park_mavic_1_viot.txt')

print('score_kcf ', 1 - np.mean( np.abs(1-occ - kcf_hog) ) )
print('score_ldes ', 1 - np.mean( np.abs(1-occ - ldes) ) )
print('score_csrdcf ', 1 - np.mean( np.abs(1-occ - csrdcf) ) )
print('score_strcf ', 1 - np.mean( np.abs(1-occ - strcf) ) )
print('score_dimp50 ', 1 - np.mean( np.abs(1-occ - dimp50) ) )
print('score_kys ', 1 - np.mean( np.abs(1-occ - kys) ) )
print('score_tomp ', 1 - np.mean( np.abs(1-occ - tomp) ) )
print('score_prdimp50 ', 1 - np.mean( np.abs(1-occ - prdimp50) ) )
print('score_mixformer ', 1 - np.mean( np.abs(1-occ - mixformer) ) )

plt.rcParams["figure.figsize"] = (20,6)
plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots()

ax.plot(range(min_idx,max_idx),1-occ, color='black', linewidth=2, label='Target \n Visibility')
# ax.plot(kcf_hog, linewidth=2, label='KCF_HOG')
# ax.plot(ldes, linewidth=2, label='LDES')
# ax.plot(csrdcf, linewidth=2, label='CSRDCF')
# ax.plot(strcf, linewidth=2, label='STRCF')
ax.plot(range(min_idx,max_idx),dimp50, linewidth=2, label='DiMP50')
ax.plot(range(min_idx,max_idx),prdimp50, linewidth=2, label='PrDiMP50')
ax.plot(range(min_idx,max_idx),kys, linewidth=2, label='KYS')
ax.plot(range(min_idx,max_idx),tomp, linewidth=2, label='ToMP')
ax.plot(range(min_idx,max_idx),mixformer, linewidth=2, label='MixFormer')


for i in range(min_idx+5, max_idx+5, 10):

    ax.axvline(x=i, ymin=0.0, ymax=1.0, color='gray', linewidth=1, linestyle='--')
    img = cv.imread('/content/pyTrackers/dataset/VIOT/park_mavic_1/{:08}.jpg'.format(i))
    
    x_min = int( np.max([gts[i, 0] - gts[i, 2], 0]) )
    x_max = int( np.min([gts[i, 0] + 2*gts[i, 2], img.shape[1]-1]) )
    y_min = int( np.max([gts[i, 1] - gts[i, 3], 0]) )
    y_max = int( np.min([gts[i, 1] + 2*gts[i, 3], img.shape[0]-1]) )

    img = img[y_min:y_max, x_min:x_max, :]
    img = cv.resize(img, (70,140))
    # cv.rectangle(img, gts[i], (0,255,255), 2)
    # binImg = cv.cvtColor(binImg, cv.COLOR_GRAY2BGR)
    img = img.astype(np.float32) / 255.0
    imagebox = OffsetImage(img, zoom=1)
    ab = AnnotationBbox(imagebox, (i, -0.9))
    ax.add_artist(ab)


# plt.title(dataset_name + '')
plt.xlabel('Frame Index')
ax.set_ylim(-1.8, 1.5)
ax.set_xlim(min_idx, max_idx + 10)
plt.ylabel('psr/psr0')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend()
plt.grid()
plt.savefig('VIOT_ratios_mixformer.pdf', format="pdf")