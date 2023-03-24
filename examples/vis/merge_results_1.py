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

import json
import numpy as np
import os
from lib.utils_ import get_ground_truthes_viot
from examples.viotdataset_config import VIOTDatasetConfig


dataset_config=VIOTDatasetConfig()

viot_results = {}

path = "../../results/"
dir_list = os.listdir(path)

with open('../all_results_viot1.json', 'r') as json_file:
    viot_results = json.load(json_file)


for file_name in dir_list:

    splitted_file_name = file_name.split("_")

    if ".txt" in file_name or ".mp4" in file_name:
        continue

    # if "viot" in file_name:
    #     continue
    if not "viot" in file_name:
        continue

    tracker = splitted_file_name[0]
    if splitted_file_name[1] == "viot":
        tracker = splitted_file_name[0] + "_" + splitted_file_name[1]

    result_json_path = path + file_name
    if os.path.isdir(result_json_path):
        continue
         
    f = open(result_json_path, 'r')
    results = json.load(f)

    for key in results.keys():

        if not key in viot_results.keys():
            viot_results[key] = {}

            gt_path = "../../dataset/VIOT/{}".format(key)
            gts = get_ground_truthes_viot(gt_path)
            start_frame,end_frame=dataset_config.frames[key][:2]
            gts = gts[start_frame - 1:end_frame]
            viot_results[key]['gts']=[]
            for gt in gts:
                viot_results[key]['gts'].append(list(gt.astype(np.int)))

        for k in results[key].keys():

            if not k in viot_results[key].keys():
                viot_results[key][k] = []

            viot_results[key][k] = results[key][k]

json_content = json.dumps(viot_results, default=str)
f = open('../all_results_mixformer_viot.json', 'w')
f.write(json_content)
f.close()