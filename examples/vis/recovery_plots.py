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
from matplotlib.ticker import MaxNLocator

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

def get_recovery(true_state, success):
    reappearance = np.where(true_state[1:] - true_state[:-1]<0)[0]
    re_count = len(reappearance)

    dts = np.zeros([5,1])
    for i in range(len(reappearance)-1):
        idx1 = reappearance[i]
        idx2 = reappearance[i+1]
        for j in range(idx1, idx2):
            if j>=len(success):
                break
            if success[j]:
                dindex = j-idx1
                if dindex<5:
                    dts[dindex] += 1
                break
    
    return dts, re_count

def draw_plot(results_json_path):
    f = open(results_json_path, 'r')
    results = json.load(f)

    num_videos=0
    dts_dimp = np.zeros([5,1])
    dts_dimpprob = np.zeros([5,1])
    dts_dimpviot = np.zeros([5,1])
    dts_dimprand = np.zeros([5,1])

    dts_motion_dimp = np.zeros([5,1])
    dts_motion_dimpprob = np.zeros([5,1])
    dts_motion_dimpviot = np.zeros([5,1])
    dts_motion_dimprand = np.zeros([5,1])

    all_re = 0
    all_mo = 0
    for data_name in results.keys():

        num_videos+=1
        data_all = results[data_name]
        gts = get_preds_by_name(data_all, 'gts')
        dimp50_preds = get_preds_by_name(data_all, 'tracker_dimp50_preds')
        dimp50_prob_preds = get_preds_by_name(data_all, 'tracker_dimp50_prob_preds')
        dimp50_viot_preds = get_preds_by_name(data_all, 'tracker_dimp50_viot_preds')
        dimp50_rand_preds = get_preds_by_name(data_all, 'tracker_dimp50_rand_preds')

        dimp50_success = get_successful_frames(gts,dimp50_preds,15)
        dimp50_prob_success = get_successful_frames(gts,dimp50_prob_preds,15)
        dimp50_viot_success = get_successful_frames(gts,dimp50_viot_preds,15)
        dimp50_rand_success = get_successful_frames(gts,dimp50_rand_preds,15)
        
        occ_path = "../../dataset/VIOT/{}/occlusion.tag".format(data_name)
        states_path = "../../dataset/VIOT/{}/camera_states.txt".format(data_name)

        occ = np.loadtxt(occ_path)
        states = np.loadtxt(states_path,delimiter=',')

        res = get_recovery(occ, dimp50_success)
        dts_dimp += res[0]
        all_re += res[1]

        res = get_recovery(occ, dimp50_prob_success)
        dts_dimpprob += res[0]

        res = get_recovery(occ, dimp50_viot_success)
        dts_dimpviot += res[0]

        res = get_recovery(occ, dimp50_rand_success)
        dts_dimprand += res[0]

        motion = (np.mean( np.abs(states[1:,4:7] - states[:-1,4:7]), axis=1 ) > 5*3.14/180).astype(np.int32)

        res = get_recovery(motion, dimp50_success)
        dts_motion_dimp += res[0]
        all_mo += res[1]

        res = get_recovery(motion, dimp50_prob_success)
        dts_motion_dimpprob += res[0]

        res = get_recovery(motion, dimp50_viot_success)
        dts_motion_dimpviot += res[0]

        res = get_recovery(motion, dimp50_rand_success)
        dts_motion_dimprand += res[0]
    
    time_step = np.mean(states[1:,0] - states[:-1,0])
    plt.plot(np.array(range(5)), 100*dts_dimp/all_re, label='DiMP50 ')
    plt.plot(np.array(range(5)), 100*dts_dimpviot/all_re, label='DiMP50_VIOT ')
    plt.plot(np.array(range(5)), 100*dts_dimpprob/all_re, label='DiMP50_PROB ')
    plt.plot(np.array(range(5)), 100*dts_dimprand/all_re, label='DiMP50_RAND ')
    plt.xlabel('Frame Count to Recovery')
    plt.ylabel('Successful Recovery (%)')
    plt.xlim([0,4])
    plt.legend()
    plt.grid()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('VIOT_recovery.pdf', format="pdf")
    plt.clf()

    print('occlusion recovery:')
    print('dimp: ',(dts_dimp/all_re)[0])
    print('dimp_prob', (dts_dimpprob/all_re)[0])
    print('dimp_viot', (dts_dimpviot/all_re)[0])
    print('dimp_rand', (dts_dimprand/all_re)[0])

    print('motion recovery:')
    print('dimp: ',(dts_motion_dimp/all_mo)[0])
    print('dimp_prob', (dts_motion_dimpprob/all_mo)[0])
    print('dimp_viot', (dts_motion_dimpviot/all_mo)[0])
    print('dimp_rand', (dts_motion_dimprand/all_mo)[0])

    # plt.plot(np.array(range(5))*time_step, dts_motion_dimp/all_mo, label='DiMP50 ')
    # plt.plot(np.array(range(5))*time_step, dts_motion_dimpprob/all_mo, label='DiMP50_PROB ')
    # plt.xlabel('Time to Recover (sec)')
    # plt.ylabel('Successful Recovery (%)')
    # plt.legend()
    # plt.grid()
    # plt.savefig('VIOT_recovery_motion.pdf', format="pdf")

if __name__=='__main__':
    result_json_path='../all_results_4.json'

    draw_plot(result_json_path)