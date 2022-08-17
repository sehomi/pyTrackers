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
from examples.vis.VIOT_info import VIOT, FREQS, FREQ_DATAS
from lib.utils import get_thresh_precision_pair,get_thresh_success_pair,calAUC

def get_preds_by_name(preds_dict,key):
    valid_keys=['gts','kcf_gray_preds','kcf_hog_preds','dcf_gray_preds',
                'dcf_hog_preds','mosse','csk','eco_hc','kcf_cn','kcf_pyECO_cn',
                'kcf_pyECO_hog','cn','DSST','DAT','Staple', 'ldes_preds',
                'strcf_preds', 'csrdcf_preds', 'tracker_dimp50_preds', 'tracker_kys_preds',
                'tracker_tomp_preds','tracker_prdimp50_preds']
    assert key in valid_keys
    str_preds=preds_dict[key]
    np_preds=[]
    for bbox in str_preds:
        bbox=[int(a) for a in bbox]
        np_preds.append(bbox)
    np_preds=np.array(np_preds)
    return np_preds

def draw_plot(datalist,dataset_name):
    plt.rcParams["figure.figsize"] = (10,6)
    plt.rcParams.update({'font.size': 16})

    # fig = plt.figure(figsize=(8, 3))
    # ax = fig.add_subplot(projection='3d')

    fig, ax = plt.subplots()

    f = open('../all_results.json', 'r')
    results = json.load(f)

    successes_kcf_gray_all = np.zeros((101,))
    successes_kcf_hog_all = np.zeros_like(successes_kcf_gray_all)
    successes_ldes_all = np.zeros_like(successes_kcf_gray_all)
    successes_strcf_all = np.zeros_like(successes_kcf_gray_all)
    successes_csrdcf_all = np.zeros_like(successes_kcf_gray_all)
    successes_dimp50_all = np.zeros_like(successes_kcf_gray_all)
    successes_prdimp50_all = np.zeros_like(successes_kcf_gray_all)
    successes_tomp_all = np.zeros_like(successes_kcf_gray_all)
    successes_kys_all = np.zeros_like(successes_kcf_gray_all)

    success_kcf_hog_all = []
    success_ldes_all = []
    success_strcf_all = []
    success_csrdcf_all = []
    success_dimp50_all = []
    success_prdimp50_all = []
    success_kys_all = []
    success_tomp_all = []

    results_list = []
    for data_name in results.keys():
        if data_name not in datalist:
            print("ignoring ", data_name)
            continue
        else:
            print("plotting ", data_name)
            results_list.append(data_name)

    num_videos=0
    for data_name in results_list:

        num_videos+=1
        data_all = results[data_name]
        gts = get_preds_by_name(data_all, 'gts')
        kcf_hog_preds = get_preds_by_name(data_all, 'kcf_hog_preds')
        ldes_preds = get_preds_by_name(data_all, 'ldes_preds')
        strcf_preds = get_preds_by_name(data_all, 'strcf_preds')
        csrdcf_preds = get_preds_by_name(data_all, 'csrdcf_preds')
        dimp50_preds = get_preds_by_name(data_all, 'tracker_dimp50_preds')
        prdimp50_preds = get_preds_by_name(data_all, 'tracker_prdimp50_preds')
        tomp_preds = get_preds_by_name(data_all, 'tracker_tomp_preds')
        kys_preds = get_preds_by_name(data_all, 'tracker_kys_preds')

        success_kcf_hog_all.append( calAUC( np.array(get_thresh_success_pair(gts, kcf_hog_preds)[1]) ) )
        success_ldes_all.append( calAUC( np.array(get_thresh_success_pair(gts, ldes_preds)[1]) ) )
        success_strcf_all.append( calAUC( np.array(get_thresh_success_pair(gts, strcf_preds)[1]) ) )
        success_csrdcf_all.append( calAUC( np.array(get_thresh_success_pair(gts, csrdcf_preds)[1]) ) )
        success_dimp50_all.append(calAUC( np.array(get_thresh_success_pair(gts, dimp50_preds)[1]) ) )
        success_prdimp50_all.append( calAUC( np.array(get_thresh_success_pair(gts, prdimp50_preds)[1]) ) )
        success_kys_all.append( calAUC( np.array(get_thresh_success_pair(gts, kys_preds)[1]) ) )
        success_tomp_all.append( calAUC( np.array(get_thresh_success_pair(gts, tomp_preds)[1]) ) )

    idxs = []
    for j in  range(len(FREQS)):
        freq = FREQS[j]
        freq_str = str(freq)

        for i in  range(len(results_list)):
            key = results_list[i]
            if freq_str in key:
                idxs.append(i)
    
    print(idxs)
    print(list(np.array(results_list)[idxs]))
    success_kcf_hog_all = np.array( success_kcf_hog_all )[idxs]
    success_ldes_all = np.array( success_ldes_all )[idxs] 
    success_strcf_all = np.array( success_strcf_all )[idxs] 
    success_csrdcf_all = np.array( success_csrdcf_all )[idxs] 
    success_dimp50_all = np.array( success_dimp50_all )[idxs] 
    success_prdimp50_all = np.array( success_prdimp50_all )[idxs] 
    success_kys_all = np.array( success_kys_all )[idxs] 
    success_tomp_all = np.array( success_tomp_all )[idxs] 


    # ax.plot(FREQS, success_kcf_hog_all, ':', color='blue', label='KCF_HOG ')
    # ax.plot(FREQS, success_ldes_all, ':', color='orange', label='LDES ')
    # ax.plot(FREQS, success_strcf_all, ':', color='green', label='STRCF ')
    # ax.plot(FREQS, success_csrdcf_all, ':', color='red', label='CSRDCF ')
    # plt.plot(FREQS, success_dimp50_all, '--', label='DiMP50 ')
    # plt.plot(FREQS, success_prdimp50_all, '--', label='PrDiMP50 ')
    # plt.plot(FREQS, success_kys_all, '--', label='KYS ')
    # plt.plot(FREQS, success_tomp_all, '--', label='ToMP ')
    # plt.title(dataset_name + '')

    f = open('../all_results_viot.json', 'r')
    results = json.load(f)

    successes_kcf_gray_all = np.zeros((101,))
    successes_kcf_hog_all = np.zeros_like(successes_kcf_gray_all)
    successes_ldes_all = np.zeros_like(successes_kcf_gray_all)
    successes_strcf_all = np.zeros_like(successes_kcf_gray_all)
    successes_csrdcf_all = np.zeros_like(successes_kcf_gray_all)
    successes_dimp50_all = np.zeros_like(successes_kcf_gray_all)
    successes_prdimp50_all = np.zeros_like(successes_kcf_gray_all)
    successes_tomp_all = np.zeros_like(successes_kcf_gray_all)
    successes_kys_all = np.zeros_like(successes_kcf_gray_all)

    success_kcf_hog_all_viot = []
    success_ldes_all_viot = []
    success_strcf_all_viot = []
    success_csrdcf_all_viot = []
    success_dimp50_all_viot = []
    success_prdimp50_all_viot = []
    success_kys_all_viot = []
    success_tomp_all_viot = []

    results_list = []
    for data_name in results.keys():
        if data_name not in datalist:
            print("ignoring ", data_name)
            continue
        else:
            print("plotting ", data_name)
            results_list.append(data_name)

    num_videos=0
    for data_name in results_list:

        num_videos+=1
        data_all = results[data_name]
        gts = get_preds_by_name(data_all, 'gts')
        kcf_hog_preds = get_preds_by_name(data_all, 'kcf_hog_preds')
        ldes_preds = get_preds_by_name(data_all, 'ldes_preds')
        strcf_preds = get_preds_by_name(data_all, 'strcf_preds')
        csrdcf_preds = get_preds_by_name(data_all, 'csrdcf_preds')
        dimp50_preds = get_preds_by_name(data_all, 'tracker_dimp50_preds')
        prdimp50_preds = get_preds_by_name(data_all, 'tracker_prdimp50_preds')
        tomp_preds = get_preds_by_name(data_all, 'tracker_tomp_preds')
        kys_preds = get_preds_by_name(data_all, 'tracker_kys_preds')

        success_kcf_hog_all_viot.append( calAUC( np.array(get_thresh_success_pair(gts, kcf_hog_preds)[1]) ) )
        success_ldes_all_viot.append( calAUC( np.array(get_thresh_success_pair(gts, ldes_preds)[1]) ) )
        success_strcf_all_viot.append( calAUC( np.array(get_thresh_success_pair(gts, strcf_preds)[1]) ) )
        success_csrdcf_all_viot.append( calAUC( np.array(get_thresh_success_pair(gts, csrdcf_preds)[1]) ) )
        success_dimp50_all_viot.append( calAUC( np.array(get_thresh_success_pair(gts, dimp50_preds)[1]) ) )
        success_prdimp50_all_viot.append( calAUC( np.array(get_thresh_success_pair(gts, prdimp50_preds)[1]) ) )
        success_kys_all_viot.append( calAUC( np.array(get_thresh_success_pair(gts, kys_preds)[1]) ) )
        success_tomp_all_viot.append( calAUC( np.array(get_thresh_success_pair(gts, tomp_preds)[1]) ) )

    idxs = []
    for j in  range(len(FREQS)):
        freq = FREQS[j]
        freq_str = str(freq)

        for i in  range(len(results_list)):
            key = results_list[i]
            if freq_str in key:
                idxs.append(i)

    success_kcf_hog_all_viot = np.array( success_kcf_hog_all_viot )[idxs]
    success_ldes_all_viot = np.array( success_ldes_all_viot )[idxs] 
    success_strcf_all_viot = np.array( success_strcf_all_viot )[idxs] 
    success_csrdcf_all_viot = np.array( success_csrdcf_all_viot )[idxs] 
    success_dimp50_all_viot = np.array( success_dimp50_all_viot )[idxs] 
    success_prdimp50_all_viot = np.array( success_prdimp50_all_viot )[idxs] 
    success_kys_all_viot = np.array( success_kys_all_viot )[idxs] 
    success_tomp_all_viot = np.array( success_tomp_all_viot )[idxs] 
 
    xs = []
    ys = []
    ys_viot = []
    for i in range(len(FREQS)):
        xs.append(FREQS[i])
        x = FREQS[i]-0.07
        y = success_kcf_hog_all[i]; ys.append(y)
        dx = 0
        dy = success_kcf_hog_all_viot[i] - y; ys_viot.append(y+dy)
        ax.plot([x], [y], 'o', color='blue')
        ax.arrow(x, y, dx, dy, head_width=0.03, head_length=0.03, linewidth=3, color='blue')

        xs.append(FREQS[i])
        x = FREQS[i]-0.035
        y = success_ldes_all[i]; ys.append(y)
        dx = 0
        dy = success_ldes_all_viot[i] - y; ys_viot.append(y+dy)
        ax.plot([x], [y], 'o', color='orange')
        ax.arrow(x, y, dx, dy, head_width=0.03, head_length=0.03, linewidth=3, color='orange')

        xs.append(FREQS[i])
        x = FREQS[i]
        y = success_strcf_all[i]; ys.append(y)
        dx = 0
        dy = success_strcf_all_viot[i] - y; ys_viot.append(y+dy)
        ax.plot([x], [y], 'o', color='green')
        ax.arrow(x, y, dx, dy, head_width=0.03, head_length=0.03, linewidth=3, color='green')

        xs.append(FREQS[i])
        x = FREQS[i]+0.035
        y = success_csrdcf_all[i]; ys.append(y)
        dx = 0
        dy = success_csrdcf_all_viot[i] - y; ys_viot.append(y+dy)
        ax.plot([x], [y], 'o', color='red')
        ax.arrow(x, y, dx, dy, head_width=0.03, head_length=0.03, linewidth=3, color='red')        
     
        # plt.plot(FREQS, success_dimp50_all, label='DiMP50_VIOT ')
        # plt.plot(FREQS, success_prdimp50_all, label='PrDiMP50_VIOT ')
        # plt.plot(FREQS, success_kys_all, label='KYS_VIOT ')
        # plt.plot(FREQS, success_tomp_all, label='ToMP_VIOT ')

    p = np.poly1d(np.polyfit(xs, ys, 1))
    p_viots = np.poly1d(np.polyfit(xs, ys_viot, 1))

    ax.plot([], [], color='red', label="CSRDCF")
    ax.plot([], [], color='green', label="STRCF")
    ax.plot([], [], color='orange', label="LDES")
    ax.plot([], [], color='blue', label="KCF")
    ax.plot(FREQS, p(FREQS), '--', color='black', linewidth=5, alpha=0.5, label="original trend")
    ax.plot(FREQS, p_viots(FREQS), color='black', linewidth=5, alpha=0.5, label="viot trend")

    for i in range(1, len(FREQS)):
        plt.axvline(x = np.mean([FREQS[i-1], FREQS[i]]), color = 'black', linewidth=0.5, alpha=0.3)

    # plt.title(dataset_name + '')
    ax.set_xlabel('Camera Motion Frequencies (Hz)')
    ax.set_ylabel('Success Rate (%)')
    ax.legend()
    ax.grid(axis='y')

    ax.set_xticks(FREQS)
    ax.set_xticklabels(FREQ_DATAS, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(dataset_name + '_freq.pdf', format="pdf")


if __name__=='__main__':

    draw_plot(FREQ_DATAS,'VIOT')