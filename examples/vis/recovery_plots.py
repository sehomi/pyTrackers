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

def get_successful_frames(gts,preds,thresh):
    length=min(len(gts),len(preds))
    gts=gts[:length,:]
    preds=preds[:length,:]
    gt_centers_x = (gts[:, 0]+gts[:,2]/2)
    gt_centers_y = (gts[:, 1]+gts[:,3]/2)
    preds_centers_x = (preds[:, 0]+preds[:,2]/2)
    preds_centers_y = (preds[:, 1]+preds[:,3]/2)
    dists = np.sqrt((gt_centers_x - preds_centers_x) ** 2 + (gt_centers_y - preds_centers_y) ** 2)
    return dists<=thresh


def draw_plot(results_json_path):
    f = open(results_json_path, 'r')
    results = json.load(f)

    num_videos=0
    dts_dimp = np.zeros([5,1])
    dts_dimpprob = np.zeros([5,1])
    all_re = 0
    for data_name in results.keys():

        num_videos+=1
        data_all = results[data_name]
        gts = get_preds_by_name(data_all, 'gts')
        dimp50_preds = get_preds_by_name(data_all, 'tracker_dimp50_preds')
        dimp50_prob_preds = get_preds_by_name(data_all, 'tracker_dimp50_prob_preds')
        # prdimp50_preds = get_preds_by_name(data_all, 'tracker_prdimp50_preds')
        # prdimp50_prob_preds = get_preds_by_name(data_all, 'tracker_prdimp50_prob_preds')

        dimp50_success = get_successful_frames(gts,dimp50_preds,15)
        dimp50_prob_success = get_successful_frames(gts,dimp50_prob_preds,15)
        
        occ_path = "../../dataset/VIOT/{}/occlusion.tag".format(data_name)
        states_path = "../../dataset/VIOT/{}/camera_states.txt".format(data_name)

        occ = np.loadtxt(occ_path)
        states = np.loadtxt(states_path,delimiter=',')

        reappearance = np.where(occ[1:] - occ[:-1]<0)[0]
        all_re += len(reappearance)

        for i in range(len(reappearance)-1):
            idx1 = reappearance[i]
            idx2 = reappearance[i+1]
            for j in range(idx1, idx2):
                if j>=len(dimp50_success):
                    break
                if dimp50_success[j]:
                    dt = states[j,0] - states[idx1,0]
                    dindex = j-idx1
                    if dindex<5:
                        # dts[int(dt*30)] += 1
                        dts_dimp[dindex] += 1
                    break

        for i in range(len(reappearance)-1):
            idx1 = reappearance[i]
            idx2 = reappearance[i+1]
            for j in range(idx1, idx2):
                if j>=len(dimp50_prob_success):
                    break
                if dimp50_prob_success[j]:
                    dt = states[j,0] - states[idx1,0]
                    dindex = j-idx1
                    if dindex<5:
                        # dts[int(dt*30)] += 1
                        dts_dimpprob[dindex] += 1
                    break
            
        # reappear = occ[1:] - occ[:-1]
        # print(data_name)
        # print(np.where(reappear<0))
        # print("****")
        # print(gts.shape)
        # print(dimp50_preds.shape)
        # print(occ.shape)
        # print(states.shape)

    plt.plot(np.array(range(5)), dts_dimp/all_re, label='DiMP50 ')
    plt.plot(np.array(range(5)), dts_dimpprob/all_re, label='DiMP50 ')
    plt.savefig('VIOT_recovery.pdf', format="pdf")

if __name__=='__main__':
    result_json_path='../all_results_4.json'

    draw_plot(result_json_path)