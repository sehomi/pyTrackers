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
import matplotlib.pyplot as plt
from examples.vis.VIOT_info import VIOT
from lib.utils import get_thresh_precision_pair,get_thresh_success_pair,calAUC


def get_preds_by_name(preds_dict,key):
    valid_keys=['gts','kcf_gray_preds','kcf_hog_preds','dcf_gray_preds',
                'dcf_hog_preds','mosse','csk','eco_hc','kcf_cn','kcf_pyECO_cn',
                'kcf_pyECO_hog','cn','DSST','DAT','Staple', 'ldes_preds',
                'strcf_preds', 'csrdcf_preds', 'tracker_dimp50_preds', 'tracker_dimp50_viot_preds',
                'tracker_dimp50_prob_preds', 'tracker_dimp50_rand_preds', 'tracker_kys_preds',
                'tracker_tomp_preds','tracker_prdimp50_preds', 'tracker_prdimp50_viot_preds',
                'tracker_prdimp50_prob_preds', 'tracker_prdimp50_rand_preds']
    assert key in valid_keys
    str_preds=preds_dict[key]
    np_preds=[]
    for bbox in str_preds:
        bbox=[int(a) for a in bbox]
        np_preds.append(bbox)
    np_preds=np.array(np_preds)
    return np_preds


def draw_plot(results_json_path,datalist,dataset_name):
    f = open(results_json_path, 'r')
    results = json.load(f)

    av_prec_dimp50 = []
    av_prec_dimp50_viot = []
    av_prec_dimp50_prob = []
    av_prec_dimp50_rand = []
    for its in range(2,2200,10):
        print(its)

        precisions_dimp50_all = np.zeros((101,))
        precisions_dimp50_viot_all = np.zeros_like(precisions_dimp50_all)
        precisions_dimp50_prob_all = np.zeros_like(precisions_dimp50_all)
        precisions_dimp50_rand_all = np.zeros_like(precisions_dimp50_all)
        precisions_prdimp50_all = np.zeros_like(precisions_dimp50_all)
        precisions_prdimp50_viot_all = np.zeros_like(precisions_dimp50_all)
        precisions_prdimp50_prob_all = np.zeros_like(precisions_dimp50_all)
        precisions_prdimp50_rand_all = np.zeros_like(precisions_dimp50_all)

        num_videos=0
        for data_name in results.keys():
            if data_name not in datalist:
                print(data_name)
                continue

            # print('len: ', len( get_preds_by_name(results[data_name], 'gts') ))
            max_idx = np.min([its, len( get_preds_by_name(results[data_name], 'gts') )])

            num_videos+=1
            data_all = results[data_name]
            gts = get_preds_by_name(data_all, 'gts')[:max_idx,:]
            dimp50_preds = get_preds_by_name(data_all, 'tracker_dimp50_preds')[:max_idx,:]
            dimp50_viot_preds = get_preds_by_name(data_all, 'tracker_dimp50_viot_preds')[:max_idx,:]
            dimp50_prob_preds = get_preds_by_name(data_all, 'tracker_dimp50_prob_preds')[:max_idx,:]
            dimp50_rand_preds = get_preds_by_name(data_all, 'tracker_dimp50_rand_preds')[:max_idx,:]
            prdimp50_preds = get_preds_by_name(data_all, 'tracker_prdimp50_preds')[:max_idx,:]
            prdimp50_viot_preds = get_preds_by_name(data_all, 'tracker_prdimp50_viot_preds')[:max_idx,:]
            prdimp50_prob_preds = get_preds_by_name(data_all, 'tracker_prdimp50_prob_preds')[:max_idx,:]
            prdimp50_rand_preds = get_preds_by_name(data_all, 'tracker_prdimp50_rand_preds')[:max_idx,:]

            precisions_dimp50_all += np.array(get_thresh_precision_pair(gts, dimp50_preds)[1])
            precisions_dimp50_viot_all += np.array(get_thresh_precision_pair(gts, dimp50_viot_preds)[1])
            precisions_dimp50_prob_all += np.array(get_thresh_precision_pair(gts, dimp50_prob_preds)[1])
            precisions_dimp50_rand_all += np.array(get_thresh_precision_pair(gts, dimp50_rand_preds)[1])
            precisions_prdimp50_all += np.array(get_thresh_precision_pair(gts, prdimp50_preds)[1])
            precisions_prdimp50_viot_all += np.array(get_thresh_precision_pair(gts, prdimp50_viot_preds)[1])
            precisions_prdimp50_prob_all += np.array(get_thresh_precision_pair(gts, prdimp50_prob_preds)[1])
            precisions_prdimp50_rand_all += np.array(get_thresh_precision_pair(gts, prdimp50_rand_preds)[1])


        precisions_dimp50_all /= num_videos
        precisions_dimp50_viot_all /= num_videos
        precisions_dimp50_prob_all /= num_videos
        precisions_dimp50_rand_all /= num_videos
        precisions_prdimp50_all /= num_videos
        precisions_prdimp50_viot_all /= num_videos
        precisions_prdimp50_prob_all /= num_videos
        precisions_prdimp50_rand_all /= num_videos


        threshes_precision = np.linspace(0, 50, 101)
        threshes_success = np.linspace(0, 1, 101)

        idx20 = [i for i, x in enumerate(threshes_precision) if x == 20][0]

        av_prec_dimp50.append(precisions_dimp50_all[idx20]*100)
        av_prec_dimp50_viot.append(precisions_dimp50_viot_all[idx20]*100)
        av_prec_dimp50_prob.append(precisions_dimp50_prob_all[idx20]*100)
        av_prec_dimp50_rand.append(precisions_dimp50_rand_all[idx20]*100)

    plt.plot(range(2,2200,10), av_prec_dimp50, label='DiMP50 ')
    plt.plot(range(2,2200,10), av_prec_dimp50_viot, label='DiMP50_VIOT ')
    plt.plot(range(2,2200,10), av_prec_dimp50_prob, label='DiMP50_PROB ')
    plt.plot(range(2,2200,10), av_prec_dimp50_rand, label='DiMP50_RAND ')
    
    plt.xlabel('Duration (frames)')
    plt.ylabel('Average Precision')
    plt.legend()
    plt.grid()
    plt.savefig(dataset_name + '_longevity.pdf', format="pdf")
    plt.clf()
    # plt.show()




if __name__=='__main__':
    result_json_path='../all_results_4.json'

    draw_plot(result_json_path,VIOT,'VIOT')
