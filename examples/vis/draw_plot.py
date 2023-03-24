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
from lib.utils_ import get_thresh_precision_pair,get_thresh_success_pair,calAUC


def get_preds_by_name(preds_dict,key):
    valid_keys=['gts','kcf_gray_preds','kcf_hog_preds','dcf_gray_preds',
                'dcf_hog_preds','mosse','csk','eco_hc','kcf_cn','kcf_pyECO_cn',
                'kcf_pyECO_hog','cn','DSST','DAT','Staple', 'ldes_preds',
                'strcf_preds', 'csrdcf_preds', 'tracker_dimp50_preds', 'tracker_kys_preds',
                'tracker_tomp_preds','tracker_prdimp50_preds', 'tracker_mixformer_preds']
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
    precisions_kcf_gray_all = np.zeros((101,))
    precisions_kcf_hog_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_ldes_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_strcf_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_csrdcf_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_dimp50_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_kys_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_prdimp50_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_tomp_all = np.zeros_like(precisions_kcf_gray_all)
    precisions_mixformer_all = np.zeros_like(precisions_kcf_gray_all)
    # precisions_dcf_gray_all = np.zeros_like(precisions_kcf_gray_all)
    # precisions_dcf_hog_all = np.zeros_like(precisions_kcf_gray_all)
    # precisions_mosse_all = np.zeros_like(precisions_kcf_gray_all)
    # precisions_csk_all = np.zeros_like(precisions_kcf_gray_all)
    # precisions_eco_hc_all = np.zeros_like(precisions_kcf_gray_all)
    # precisions_kcf_cn_all=np.zeros_like(precisions_kcf_gray_all)
    # precisions_kcf_pyECO_cn_all=np.zeros_like(precisions_kcf_gray_all)
    # precisions_kcf_pyECO_hog_all=np.zeros_like(precisions_kcf_gray_all)
    # precisions_cn_all=np.zeros_like(precisions_kcf_gray_all)
    # precisions_dsst_all=np.zeros_like(precisions_kcf_gray_all)
    # precisions_dat_all=np.zeros_like(precisions_kcf_gray_all)
    # precisions_staple_all=np.zeros_like(precisions_kcf_gray_all)

    successes_kcf_gray_all = np.zeros((101,))
    successes_kcf_hog_all = np.zeros_like(successes_kcf_gray_all)
    successes_ldes_all = np.zeros_like(successes_kcf_gray_all)
    successes_strcf_all = np.zeros_like(successes_kcf_gray_all)
    successes_csrdcf_all = np.zeros_like(successes_kcf_gray_all)
    successes_dimp50_all = np.zeros_like(successes_kcf_gray_all)
    successes_prdimp50_all = np.zeros_like(successes_kcf_gray_all)
    successes_tomp_all = np.zeros_like(successes_kcf_gray_all)
    successes_kys_all = np.zeros_like(successes_kcf_gray_all)
    successes_mixformer_all = np.zeros_like(successes_kcf_gray_all)
    # successes_dcf_gray_all = np.zeros_like(successes_kcf_gray_all)
    # successes_dcf_hog_all = np.zeros_like(successes_kcf_gray_all)
    # successes_mosse_all = np.zeros_like(successes_kcf_gray_all)
    # successes_csk_all = np.zeros_like(successes_kcf_gray_all)
    # successes_eco_hc_all = np.zeros_like(successes_kcf_gray_all)
    # successes_kcf_cn_all=np.zeros_like(successes_kcf_gray_all)
    # successes_kcf_pyECO_cn_all=np.zeros_like(successes_kcf_gray_all)
    # successes_kcf_pyECO_hog_all=np.zeros_like(successes_kcf_gray_all)
    # successes_cn_all=np.zeros_like(successes_kcf_gray_all)
    # successes_dsst_all=np.zeros_like(successes_kcf_gray_all)
    # successes_dat_all=np.zeros_like(successes_kcf_gray_all)
    # successes_staple_all=np.zeros_like(successes_kcf_gray_all)
    num_videos=0
    for data_name in results.keys():
        if data_name not in datalist:
            print(data_name)
            continue

        num_videos+=1
        data_all = results[data_name]
        gts = get_preds_by_name(data_all, 'gts')
        # kcf_gray_preds = get_preds_by_name(data_all, 'kcf_gray_preds')
        kcf_hog_preds = get_preds_by_name(data_all, 'kcf_hog_preds')
        ldes_preds = get_preds_by_name(data_all, 'ldes_preds')
        strcf_preds = get_preds_by_name(data_all, 'strcf_preds')
        csrdcf_preds = get_preds_by_name(data_all, 'csrdcf_preds')
        dimp50_preds = get_preds_by_name(data_all, 'tracker_dimp50_preds')
        prdimp50_preds = get_preds_by_name(data_all, 'tracker_prdimp50_preds')
        tomp_preds = get_preds_by_name(data_all, 'tracker_tomp_preds')
        kys_preds = get_preds_by_name(data_all, 'tracker_kys_preds')
        mixformer_preds = get_preds_by_name(data_all, 'tracker_mixformer_preds')
        # dcf_gray_preds = get_preds_by_name(data_all, 'dcf_gray_preds')
        # dcf_hog_preds = get_preds_by_name(data_all, 'dcf_hog_preds')
        # mosse_preds = get_preds_by_name(data_all, 'mosse')
        # csk_preds = get_preds_by_name(data_all, 'csk')
        # eco_hc_preds = get_preds_by_name(data_all, 'eco_hc')
        # kcf_cn_preds=get_preds_by_name(data_all,'kcf_cn')
        # kcf_pyECO_cn_preds=get_preds_by_name(data_all,'kcf_pyECO_cn')
        # kcf_pyECO_hog_preds=get_preds_by_name(data_all,'kcf_pyECO_hog')
        # cn_preds=get_preds_by_name(data_all,'cn')
        # dsst_preds=get_preds_by_name(data_all,'DSST')
        # dat_preds=get_preds_by_name(data_all,'DAT')
        # staple_preds=get_preds_by_name(data_all,'Staple')

        # precisions_kcf_gray_all += np.array(get_thresh_precision_pair(gts, kcf_gray_preds)[1])
        precisions_kcf_hog_all += np.array(get_thresh_precision_pair(gts, kcf_hog_preds)[1])
        precisions_ldes_all += np.array(get_thresh_precision_pair(gts, ldes_preds)[1])
        precisions_strcf_all += np.array(get_thresh_precision_pair(gts, strcf_preds)[1])
        precisions_csrdcf_all += np.array(get_thresh_precision_pair(gts, csrdcf_preds)[1])
        precisions_dimp50_all += np.array(get_thresh_precision_pair(gts, dimp50_preds)[1])
        precisions_prdimp50_all += np.array(get_thresh_precision_pair(gts, prdimp50_preds)[1])
        precisions_tomp_all += np.array(get_thresh_precision_pair(gts, tomp_preds)[1])
        precisions_kys_all += np.array(get_thresh_precision_pair(gts, kys_preds)[1])
        precisions_mixformer_all += np.array(get_thresh_precision_pair(gts, mixformer_preds)[1])
        # precisions_dcf_gray_all += np.array(get_thresh_precision_pair(gts, dcf_gray_preds)[1])
        # precisions_dcf_hog_all += np.array(get_thresh_precision_pair(gts, dcf_hog_preds)[1])
        # precisions_mosse_all += np.array(get_thresh_precision_pair(gts, mosse_preds)[1])
        # precisions_csk_all += np.array(get_thresh_precision_pair(gts, csk_preds)[1])
        # precisions_eco_hc_all += np.array(get_thresh_precision_pair(gts, eco_hc_preds)[1])
        # precisions_kcf_cn_all+=np.array(get_thresh_precision_pair(gts,kcf_cn_preds)[1])
        # precisions_kcf_pyECO_cn_all+=np.array(get_thresh_precision_pair(gts,kcf_pyECO_cn_preds)[1])
        # precisions_kcf_pyECO_hog_all+=np.array(get_thresh_precision_pair(gts,kcf_pyECO_hog_preds)[1])
        # precisions_cn_all+=np.array(get_thresh_precision_pair(gts,cn_preds)[1])
        # precisions_dsst_all+=np.array(get_thresh_precision_pair(gts,dsst_preds)[1])
        # precisions_dat_all+=np.array(get_thresh_precision_pair(gts,dat_preds)[1])
        # precisions_staple_all+=np.array(get_thresh_precision_pair(gts,staple_preds)[1])

        # successes_kcf_gray_all += np.array(get_thresh_success_pair(gts, kcf_gray_preds)[1])
        successes_kcf_hog_all += np.array(get_thresh_success_pair(gts, kcf_hog_preds)[1])
        successes_ldes_all += np.array(get_thresh_success_pair(gts, ldes_preds)[1])
        successes_strcf_all += np.array(get_thresh_success_pair(gts, strcf_preds)[1])
        successes_csrdcf_all += np.array(get_thresh_success_pair(gts, csrdcf_preds)[1])
        successes_dimp50_all += np.array(get_thresh_success_pair(gts, dimp50_preds)[1])
        successes_prdimp50_all += np.array(get_thresh_success_pair(gts, prdimp50_preds)[1])
        successes_kys_all += np.array(get_thresh_success_pair(gts, kys_preds)[1])
        successes_tomp_all += np.array(get_thresh_success_pair(gts, tomp_preds)[1])
        successes_mixformer_all += np.array(get_thresh_success_pair(gts, mixformer_preds)[1])
        # successes_dcf_gray_all += np.array(get_thresh_success_pair(gts, dcf_gray_preds)[1])
        # successes_dcf_hog_all += np.array(get_thresh_success_pair(gts, dcf_hog_preds)[1])
        # successes_mosse_all += np.array(get_thresh_success_pair(gts, mosse_preds)[1])
        # successes_csk_all += np.array(get_thresh_success_pair(gts, csk_preds)[1])
        # successes_eco_hc_all += np.array(get_thresh_success_pair(gts, eco_hc_preds)[1])
        # successes_kcf_cn_all+=np.array(get_thresh_success_pair(gts,kcf_cn_preds)[1])
        # successes_kcf_pyECO_cn_all+=np.array(get_thresh_success_pair(gts,kcf_pyECO_cn_preds)[1])
        # successes_kcf_pyECO_hog_all+=np.array(get_thresh_success_pair(gts,kcf_pyECO_hog_preds)[1])
        # successes_cn_all+=np.array(get_thresh_success_pair(gts,cn_preds)[1])
        # successes_dsst_all+=np.array(get_thresh_success_pair(gts,dsst_preds)[1])
        # successes_dat_all+=np.array(get_thresh_success_pair(gts,dat_preds)[1])
        # successes_staple_all+=np.array(get_thresh_success_pair(gts,staple_preds)[1])

    # precisions_kcf_gray_all /= num_videos
    precisions_kcf_hog_all /= num_videos
    precisions_ldes_all /= num_videos
    precisions_strcf_all /= num_videos
    precisions_csrdcf_all /= num_videos
    precisions_dimp50_all /= num_videos
    precisions_prdimp50_all /= num_videos
    precisions_kys_all /= num_videos
    precisions_tomp_all /= num_videos
    precisions_mixformer_all /= num_videos
    # precisions_dcf_gray_all /= num_videos
    # precisions_dcf_hog_all /= num_videos
    # precisions_csk_all /= num_videos
    # precisions_mosse_all /= num_videos
    # precisions_eco_hc_all /= num_videos
    # precisions_kcf_cn_all/=num_videos
    # precisions_kcf_pyECO_cn_all/=num_videos
    # precisions_kcf_pyECO_hog_all/=num_videos
    # precisions_cn_all/=num_videos
    # precisions_dsst_all/=num_videos
    # precisions_dat_all/=num_videos
    # precisions_staple_all/=num_videos

    # successes_kcf_gray_all /= num_videos
    successes_kcf_hog_all /= num_videos
    successes_ldes_all /= num_videos
    successes_strcf_all /= num_videos
    successes_csrdcf_all /= num_videos
    successes_dimp50_all /= num_videos
    successes_prdimp50_all /= num_videos
    successes_kys_all /= num_videos
    successes_tomp_all /= num_videos
    successes_mixformer_all /= num_videos
    # successes_dcf_gray_all /= num_videos
    # successes_dcf_hog_all /= num_videos
    # successes_csk_all /= num_videos
    # successes_mosse_all /= num_videos
    # successes_eco_hc_all /= num_videos
    # successes_kcf_cn_all/=num_videos
    # successes_kcf_pyECO_cn_all/=num_videos
    # successes_kcf_pyECO_hog_all/=num_videos
    # successes_cn_all/=num_videos
    # successes_dsst_all/=num_videos
    # successes_dat_all/=num_videos
    # successes_staple_all/=num_videos

    threshes_precision = np.linspace(0, 50, 101)
    threshes_success = np.linspace(0, 1, 101)

    idx20 = [i for i, x in enumerate(threshes_precision) if x == 20][0]
    # plt.plot(threshes_precision, precisions_eco_hc_all, label='ECO-HC ' + str(precisions_eco_hc_all[idx20])[:5])
    #plt.plot(threshes_precision, precisions_kcf_gray_all, label='KCF_GRAY ' + str(precisions_kcf_gray_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_kcf_hog_all, label='KCF_HOG ' + str(precisions_kcf_hog_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_ldes_all, label='LDES ' + str(precisions_ldes_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_strcf_all, label='STRCF ' + str(precisions_strcf_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_csrdcf_all, label='CSRDCF ' + str(precisions_csrdcf_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_dimp50_all, label='DiMP50 ' + str(precisions_dimp50_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_prdimp50_all, label='PrDiMP50 ' + str(precisions_prdimp50_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_kys_all, label='KYS ' + str(precisions_kys_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_tomp_all, label='ToMP ' + str(precisions_tomp_all[idx20])[:5])
    plt.plot(threshes_precision, precisions_mixformer_all, label='MixFormer ' + str(precisions_mixformer_all[idx20])[:5])
    #plt.plot(threshes_precision, precisions_dcf_gray_all, label='DCF_GRAY ' + str(precisions_dcf_gray_all[idx20])[:5])
    # plt.plot(threshes_precision, precisions_dcf_hog_all, label='DCF_HOG ' + str(precisions_dcf_hog_all[idx20])[:5])
    # plt.plot(threshes_precision, precisions_mosse_all, label='MOSSE ' + str(precisions_mosse_all[idx20])[:5])
    # plt.plot(threshes_precision, precisions_csk_all, label='CSK ' + str(precisions_csk_all[idx20])[:5])
    #plt.plot(threshes_precision,precisions_kcf_cn_all,label='KCF_CN '+str(precisions_kcf_cn_all[idx20])[:5])
    #plt.plot(threshes_precision,precisions_kcf_pyECO_cn_all,label='KCF_pyECO_CN '+str(precisions_kcf_pyECO_cn_all[idx20])[:5])
    #plt.plot(threshes_precision,precisions_kcf_pyECO_hog_all,label='KCF_pyECO_HOG '+str(precisions_kcf_pyECO_hog_all[idx20])[:5])
    #plt.plot(threshes_precision,precisions_cn_all,label='CN '+str(precisions_cn_all[idx20])[:5])
    # plt.plot(threshes_precision,precisions_dsst_all,label='DSST '+str(precisions_dsst_all[idx20])[:5])
    # plt.plot(threshes_precision,precisions_dat_all,label='DAT '+str(precisions_dat_all[idx20])[:5])
    # plt.plot(threshes_precision,precisions_staple_all,label='Staple '+str(precisions_staple_all[idx20])[:5])
    plt.xlabel('Location error threshold')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.savefig(dataset_name + '_precision.pdf', format="pdf")
    plt.clf()
    # plt.show()

    # plt.plot(threshes_success, successes_eco_hc_all, label='ECO-HC' + str(calAUC(successes_eco_hc_all))[:5])
    #plt.plot(threshes_success, successes_kcf_gray_all, label='KCF_GRAY ' + str(calAUC(successes_kcf_gray_all))[:5])
    plt.plot(threshes_success, successes_kcf_hog_all, label='KCF_HOG ' + str(calAUC(successes_kcf_hog_all))[:5])
    plt.plot(threshes_success, successes_ldes_all, label='LDES ' + str(calAUC(successes_ldes_all))[:5])
    plt.plot(threshes_success, successes_strcf_all, label='STRCF ' + str(calAUC(successes_strcf_all))[:5])
    plt.plot(threshes_success, successes_csrdcf_all, label='CSRDCF ' + str(calAUC(successes_csrdcf_all))[:5])
    plt.plot(threshes_success, successes_dimp50_all, label='DiMP50 ' + str(calAUC(successes_dimp50_all))[:5])
    plt.plot(threshes_success, successes_prdimp50_all, label='PrDiMP50 ' + str(calAUC(successes_prdimp50_all))[:5])
    plt.plot(threshes_success, successes_kys_all, label='KYS ' + str(calAUC(successes_kys_all))[:5])
    plt.plot(threshes_success, successes_tomp_all, label='ToMP ' + str(calAUC(successes_tomp_all))[:5])
    plt.plot(threshes_success, successes_mixformer_all, label='MixFormer ' + str(calAUC(successes_mixformer_all))[:5])
    #plt.plot(threshes_success, successes_dcf_gray_all, label='DCF_GRAY ' + str(calAUC(successes_dcf_gray_all))[:5])
    # plt.plot(threshes_success, successes_dcf_hog_all, label='DCF_HOG ' + str(calAUC(successes_dcf_hog_all))[:5])
    # plt.plot(threshes_success, successes_mosse_all, label='MOSSE ' + str(calAUC(successes_mosse_all))[:5])
    # plt.plot(threshes_success, successes_csk_all, label='CSK ' + str(calAUC(successes_csk_all))[:5])
    #plt.plot(threshes_success,successes_kcf_cn_all,label='KCF_CN '+str(calAUC(successes_kcf_cn_all))[:5])
    #plt.plot(threshes_success,successes_kcf_pyECO_cn_all,label='KCF_pyECO_CN '+str(calAUC(successes_kcf_pyECO_cn_all))[:5])
    #plt.plot(threshes_success,successes_kcf_pyECO_hog_all,label='KCF_pyECO_HOG '+str(calAUC(successes_kcf_pyECO_hog_all))[:5])
    #plt.plot(threshes_success,successes_cn_all,label='CN '+str(calAUC(successes_cn_all))[:5])
    # plt.plot(threshes_success,successes_dsst_all,label='DSST '+str(calAUC(successes_dsst_all))[:5])
    # plt.plot(threshes_success,successes_dat_all,label='DAT '+str(calAUC(successes_dat_all))[:5])
    # plt.plot(threshes_success,successes_staple_all,label='Staple '+str(calAUC(successes_staple_all))[:5])
    plt.xlabel('Overlap Threshold')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid()
    plt.savefig(dataset_name + '_success.pdf', format="pdf")
    plt.clf()
    print(dataset_name,':',num_videos)


if __name__=='__main__':
    result_json_path='../all_results_mixformer_viot.json'

    draw_plot(result_json_path,VIOT,'NEW')
