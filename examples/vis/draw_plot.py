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
                'tracker_dimp50_prob_preds', 'tracker_kys_preds',
                'tracker_tomp_preds','tracker_prdimp50_preds', 'tracker_prdimp50_viot_preds',
                'tracker_prdimp50_prob_preds']
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
    precisions_dimp50_all = np.zeros((101,))
    precisions_dimp50_viot_all = np.zeros_like(precisions_dimp50_all)
    precisions_dimp50_prob_all = np.zeros_like(precisions_dimp50_all)
    precisions_prdimp50_all = np.zeros_like(precisions_dimp50_all)
    precisions_prdimp50_viot_all = np.zeros_like(precisions_dimp50_all)
    precisions_prdimp50_prob_all = np.zeros_like(precisions_dimp50_all)

    successes_dimp50_all = np.zeros((101,))
    successes_dimp50_viot_all = np.zeros_like(successes_dimp50_all)
    successes_dimp50_prob_all = np.zeros_like(successes_dimp50_all)
    successes_prdimp50_all = np.zeros_like(successes_dimp50_all)
    successes_prdimp50_viot_all = np.zeros_like(successes_dimp50_all)
    successes_prdimp50_prob_all = np.zeros_like(successes_dimp50_all)

    num_videos=0
    for data_name in results.keys():
        if data_name not in datalist:
            print(data_name)
            continue

        num_videos+=1
        data_all = results[data_name]
        gts = get_preds_by_name(data_all, 'gts')
        dimp50_preds = get_preds_by_name(data_all, 'tracker_dimp50_preds')
        dimp50_viot_preds = get_preds_by_name(data_all, 'tracker_dimp50_viot_preds')
        dimp50_prob_preds = get_preds_by_name(data_all, 'tracker_dimp50_prob_preds')
        prdimp50_preds = get_preds_by_name(data_all, 'tracker_prdimp50_preds')
        prdimp50_viot_preds = get_preds_by_name(data_all, 'tracker_prdimp50_viot_preds')
        prdimp50_prob_preds = get_preds_by_name(data_all, 'tracker_prdimp50_prob_preds')

        precisions_dimp50_all += np.array(get_thresh_precision_pair(gts, dimp50_preds)[1])
        precisions_dimp50_viot_all += np.array(get_thresh_precision_pair(gts, dimp50_viot_preds)[1])
        precisions_dimp50_prob_all += np.array(get_thresh_precision_pair(gts, dimp50_prob_preds)[1])
        precisions_prdimp50_all += np.array(get_thresh_precision_pair(gts, prdimp50_preds)[1])
        precisions_prdimp50_viot_all += np.array(get_thresh_precision_pair(gts, prdimp50_viot_preds)[1])
        precisions_prdimp50_prob_all += np.array(get_thresh_precision_pair(gts, prdimp50_prob_preds)[1])

        successes_dimp50_all += np.array(get_thresh_success_pair(gts, dimp50_preds)[1])
        successes_dimp50_viot_all += np.array(get_thresh_success_pair(gts, dimp50_viot_preds)[1])
        successes_dimp50_prob_all += np.array(get_thresh_success_pair(gts, dimp50_prob_preds)[1])
        successes_prdimp50_all += np.array(get_thresh_success_pair(gts, prdimp50_preds)[1])
        successes_prdimp50_viot_all += np.array(get_thresh_success_pair(gts, prdimp50_viot_preds)[1])
        successes_prdimp50_prob_all += np.array(get_thresh_success_pair(gts, prdimp50_prob_preds)[1])



    precisions_dimp50_all /= num_videos
    precisions_dimp50_viot_all /= num_videos
    precisions_dimp50_prob_all /= num_videos
    precisions_prdimp50_all /= num_videos
    precisions_prdimp50_viot_all /= num_videos
    precisions_prdimp50_prob_all /= num_videos

    successes_dimp50_all /= num_videos
    successes_dimp50_viot_all /= num_videos
    successes_dimp50_prob_all /= num_videos
    successes_prdimp50_all /= num_videos
    successes_prdimp50_viot_all /= num_videos
    successes_prdimp50_prob_all /= num_videos

    threshes_precision = np.linspace(0, 50, 101)
    threshes_success = np.linspace(0, 1, 101)

    idx20 = [i for i, x in enumerate(threshes_precision) if x == 20][0]

    plt.plot(threshes_precision, precisions_dimp50_all, label='DiMP50 ' + str(precisions_dimp50_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_dimp50_viot_all, label='DiMP50_VIOT ' + str(precisions_dimp50_viot_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_dimp50_prob_all, label='DiMP50_PROB ' + str(precisions_dimp50_prob_all[idx20])[:5])
    # plt.plot(threshes_precision, precisions_prdimp50_all, label='PrDiMP50 ' + str(precisions_prdimp50_all[idx20])[:5])
    # plt.plot(threshes_precision, precisions_prdimp50_viot_all, label='PrDiMP50_VIOT ' + str(precisions_prdimp50_viot_all[idx20])[:5])
    # plt.plot(threshes_precision, precisions_prdimp50_prob_all, label='PrDiMP50_PROB ' + str(precisions_prdimp50_prob_all[idx20])[:5])
    
    plt.xlabel('Location error threshold')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.savefig(dataset_name + '_precision3.pdf', format="pdf")
    plt.clf()
    # plt.show()

    plt.plot(threshes_success, successes_dimp50_all, label='DiMP50 ' + str(calAUC(successes_dimp50_all))[:5])
    plt.plot(threshes_success, successes_dimp50_viot_all, label='DiMP50_VIOT ' + str(calAUC(successes_dimp50_viot_all))[:5])
    plt.plot(threshes_success, successes_dimp50_prob_all, label='DiMP50_PROB ' + str(calAUC(successes_dimp50_prob_all))[:5])
    # plt.plot(threshes_success, successes_prdimp50_all, label='PrDiMP50 ' + str(calAUC(successes_prdimp50_all))[:5])
    # plt.plot(threshes_success, successes_prdimp50_viot_all, label='PrDiMP50_VIOT ' + str(calAUC(successes_prdimp50_viot_all))[:5])
    # plt.plot(threshes_success, successes_prdimp50_prob_all, label='PrDiMP50_PROB ' + str(calAUC(successes_prdimp50_prob_all))[:5])
   
    plt.xlabel('Overlap Threshold')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid()
    plt.savefig(dataset_name + '_success3.pdf', format="pdf")
    plt.clf()
    print(dataset_name,':',num_videos)


if __name__=='__main__':
    result_json_path='../all_results_3.json'

    draw_plot(result_json_path,VIOT,'VIOT')
