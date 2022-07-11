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


        # tracker_kcf_hog=PyTracker(img_dir,tracker_type='KCF_HOG',dataset_config=dataset_config)
        # tracker_ldes=PyTracker(img_dir,tracker_type='LDES',dataset_config=dataset_config)
        # tracker_strcf=PyTracker(img_dir,tracker_type='STRCF',dataset_config=dataset_config)
        # tracker_csrdcf=PyTracker(img_dir,tracker_type='CSRDCF',dataset_config=dataset_config)
        # tracker_eco=PyTracker(img_dir,tracker_type='ECO',dataset_config=dataset_config)
        # tracker_prdimp50=PyTracker(img_dir,tracker_type='PRDIMP50',dataset_config=dataset_config)
        # tracker_kys=PyTracker(img_dir,tracker_type='KYS',dataset_config=dataset_config)
        # tracker_tomp=PyTracker(img_dir,tracker_type='TOMP',dataset_config=dataset_config)
        tracker_dimp50=PyTracker(img_dir,tracker_type='DIMP50',dataset_config=dataset_config)

        dimp50_preds=tracker_dimp50.tracking(verbose=True,video_path="../results/dimp50_{:s}.mp4".format(data_name))
        dimp50_results = {}
        dimp50_results[data_name] = {}
        dimp50_results[data_name]['tracker_dimp50_preds'] = []
        for dimp50_pred in dimp50_preds:
            dimp50_results[data_name]['tracker_dimp50_preds'].append(list(dimp50_pred.astype(np.int)))
        write_results(data_name, 'dimp50', dimp50_results)
        print('dimp50 done!')
        
        # kys_preds=tracker_kys.tracking(verbose=True,video_path="../results/kys_{:s}.mp4".format(data_name))
        # kys_results = {}
        # kys_results[data_name] = {}
        # kys_results[data_name]['tracker_kys_preds'] = []
        # for kys_pred in kys_preds:
        #     kys_results[data_name]['tracker_kys_preds'].append(list(kys_pred.astype(np.int)))
        # write_results(data_name, 'kys', kys_results)
        # print('kys done!')

        # tomp_preds=tracker_tomp.tracking(verbose=True,video_path="../results/tomp_{:s}.mp4".format(data_name))
        # tomp_results = {}
        # tomp_results[data_name] = {}
        # tomp_results[data_name]['tracker_tomp_preds'] = []
        # for tomp_pred in tomp_preds:
        #     tomp_results[data_name]['tracker_tomp_preds'].append(list(tomp_pred.astype(np.int)))
        # write_results(data_name, 'tomp', tomp_results)
        # print('tomp done!')

        # prdimp50_preds=tracker_prdimp50.tracking(verbose=True,video_path="../results/prdimp50_{:s}.mp4".format(data_name))
        # prdimp50_results = {}
        # prdimp50_results[data_name] = {}
        # prdimp50_results[data_name]['tracker_prdimp50_preds'] = []
        # for prdimp50_pred in prdimp50_preds:
        #     prdimp50_results[data_name]['tracker_prdimp50_preds'].append(list(prdimp50_pred.astype(np.int)))
        # write_results(data_name, 'prdimp50', prdimp50_results)
        # print('prdimp50 done!')

        # # eco_preds = tracker_eco.tracking()
        # # eco_results = {}
        # # eco_results[data_name] = {}
        # # eco_results[data_name]['eco'] = []
        # # for eco_pred in eco_preds:
        # #     eco_results[data_name]['eco'].append(list(eco_pred.astype(np.int)))
        # # write_results(data_name, 'eco', eco_results)
        # # print('eco done!')

        # kcf_hog_preds=tracker_kcf_hog.tracking(verbose=True,video_path="../results/kcf_{:s}.mp4".format(data_name))
        # kcf_hog_results = {}
        # kcf_hog_results[data_name] = {}
        # kcf_hog_results[data_name]['kcf_hog_preds'] = []
        # for kcf_hog_pred in kcf_hog_preds:
        #     kcf_hog_results[data_name]['kcf_hog_preds'].append(list(kcf_hog_pred.astype(np.int)))
        # write_results(data_name, 'kcf_hog', kcf_hog_results)
        # print('kcf hog done!')

        # ldes_preds=tracker_ldes.tracking(verbose=True,video_path="../results/ldes_{:s}.mp4".format(data_name))
        # ldes_results = {}
        # ldes_results[data_name] = {}
        # ldes_results[data_name]['ldes_preds'] = []
        # for ldes_pred in ldes_preds:
        #     ldes_results[data_name]['ldes_preds'].append(list(ldes_pred.astype(np.int)))
        # write_results(data_name, 'ldes', ldes_results)
        # print('ldes done!')

        # csrdcf_preds=tracker_csrdcf.tracking(verbose=True,video_path="../results/csrdcf_{:s}.mp4".format(data_name))
        # csrdcf_results = {}
        # csrdcf_results[data_name] = {}
        # csrdcf_results[data_name]['csrdcf_preds'] = []
        # for csrdcf_pred in csrdcf_preds:
        #     csrdcf_results[data_name]['csrdcf_preds'].append(list(csrdcf_pred.astype(np.int)))
        # write_results(data_name, 'csrdcf', csrdcf_results)
        # print('csrdcf done!')

        # strcf_preds=tracker_strcf.tracking(verbose=True,video_path="../results/strcf_{:s}.mp4".format(data_name))
        # strcf_results = {}
        # strcf_results[data_name] = {}
        # strcf_results[data_name]['strcf_preds'] = []
        # for strcf_pred in strcf_preds:
        #     strcf_results[data_name]['strcf_preds'].append(list(strcf_pred.astype(np.int)))
        # write_results(data_name, 'strcf', strcf_results)
        # print('strcf done!')

