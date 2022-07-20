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

import os
import numpy as np
import matplotlib.pyplot as plt
from examples.pytracker import PyTracker
import json
from lib.utils import get_ground_truthes,get_ground_truthes_viot,get_thresh_success_pair,get_thresh_precision_pair,calAUC
from examples.viotdataset_config import VIOTDatasetConfig

def write_results(data_name, tracker, results):
    json_content = json.dumps(results, default=str)
    f = open('../results/{:s}_{:s}.json'.format(tracker, data_name), 'w')
    f.write(json_content)
    f.close()

def mode_to_str(mode):
    txt = ''
    if mode in ['viot', 'prob']:
        txt = '_{}'.format(mode)

    return txt


if __name__ == '__main__':
    # data_dir = '../dataset/OTB100'
    data_dir = '../dataset/VIOT'
    # data_names=sorted(os.listdir(data_dir)) ## OTB
    data_names=sorted([ name for name in os.listdir(data_dir) if \
                        os.path.isdir(os.path.join(data_dir, name)) ]) ## VIOT
    viot_results = {}

    # dataset_config=OTBDatasetConfig()
    dataset_config=VIOTDatasetConfig()

    for data_name in data_names:

        print('data name:', data_name)
        viot_results[data_name]={}
        data_path = os.path.join(data_dir, data_name)
        # img_dir = os.path.join(data_path,'img') ## OTB
        img_dir = data_path ## VIOT

        gts=get_ground_truthes_viot(data_path)
        if data_name in dataset_config.frames.keys():
            start_frame,end_frame=dataset_config.frames[data_name][:2]
            gts = gts[start_frame - 1:end_frame]
        else:
            continue

        viot_results[data_name]['gts']=[]
        for gt in gts:
            viot_results[data_name]['gts'].append(list(gt.astype(np.int)))


        tracker_prdimp50=PyTracker(img_dir,tracker_type='PRDIMP50',dataset_config=dataset_config)
        tracker_dimp50=PyTracker(img_dir,tracker_type='DIMP50',dataset_config=dataset_config)

        method_str = mode_to_str(tracker_dimp50._kin_configs.configs['method'])
        dimp50_preds=tracker_dimp50.tracking(verbose=True,video_path="../results/dimp50{:s}_{:s}.mp4".format(method_str, data_name))
        dimp50_results = {}
        dimp50_results[data_name] = {}
        dimp50_results[data_name]['tracker_dimp50{:s}_preds'.format(method_str)] = []
        for dimp50_pred in dimp50_preds:
            dimp50_results[data_name]['tracker_dimp50{:s}_preds'.format(method_str)].append(list(dimp50_pred.astype(np.int)))
        write_results(data_name, 'dimp50{:s}'.format(method_str), dimp50_results)
        print('dimp50 done!')
        
        method_str = mode_to_str(tracker_prdimp50._kin_configs.configs['method'])
        prdimp50_preds=tracker_prdimp50.tracking(verbose=True,video_path="../results/prdimp50{:s}_{:s}.mp4".format(method_str, data_name))
        prdimp50_results = {}
        prdimp50_results[data_name] = {}
        prdimp50_results[data_name]['tracker_prdimp50{:s}_preds'.format(method_str)] = []
        for prdimp50_pred in prdimp50_preds:
            prdimp50_results[data_name]['tracker_prdimp50{:s}_preds'.format(method_str)].append(list(prdimp50_pred.astype(np.int)))
        write_results(data_name, 'prdimp50{:s}'.format(method_str), prdimp50_results)
        print('prdimp50 done!')

