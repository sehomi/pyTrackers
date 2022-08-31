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
import cv2
from lib.utils import get_img_list
from examples.vis.draw_plot import get_preds_by_name
from lib.utils import get_thresh_success_pair,calAUC
from examples.viotdataset_config import VIOTDatasetConfig

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=8):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(img,pts,color,thickness,style)
    return img


def vis_results(dataset_dir,data_name):
    dataset_config=VIOTDatasetConfig()
    f = open('../all_results_4.json', 'r')
    results = json.load(f)
    if not data_name in results.keys():
        raise ValueError
    # cap=cv2.VideoCapture(data_name+'_vis.avi')
    data_all = results[data_name]

    gts = get_preds_by_name(data_all, 'gts')
    dimp50_preds = get_preds_by_name(data_all, 'tracker_dimp50_preds')
    dimp50_prob_preds = get_preds_by_name(data_all, 'tracker_dimp50_prob_preds')
    dimp50_viot_preds = get_preds_by_name(data_all, 'tracker_dimp50_viot_preds')
    dimp50_rand_preds = get_preds_by_name(data_all, 'tracker_dimp50_rand_preds')


    img_dir=os.path.join(dataset_dir,data_name)

    img_list=get_img_list(img_dir)
    start_frame,end_frame=dataset_config.frames[data_name][0:2]
    img_list=img_list[start_frame-1:end_frame]
    imgs = []

    color_dimp50=(255,0,0) #
    color_dimp50_viot=(0,125,255) #
    color_dimp50_prob=(0,255,0) #
    color_dimp50_rand=(0,0,255) #


    writer=None
    for i in range(len(img_list)):

        # flag,current_frame=cap.read()
        # if flag is False:
        #     return

        current_frame = cv2.imread(img_list[i])
        show_frame = current_frame.copy()
        gt=gts[i]
        dimp50_pred = dimp50_preds[i]
        dimp50_prob_pred = dimp50_prob_preds[i]
        dimp50_viot_pred = dimp50_viot_preds[i]
        dimp50_rand_pred = dimp50_rand_preds[i]
        
        show_frame=cv2.rectangle(show_frame,(dimp50_pred[0],dimp50_pred[1]),
                                 (dimp50_pred[0]+dimp50_pred[2],dimp50_pred[1]+dimp50_pred[3]),
                                 color_dimp50,thickness=2)
        show_frame=cv2.rectangle(show_frame,(dimp50_prob_pred[0],dimp50_prob_pred[1]),
                                 (dimp50_prob_pred[0]+dimp50_prob_pred[2],dimp50_prob_pred[1]+dimp50_prob_pred[3]),
                                 color_dimp50_prob,thickness=2)
        show_frame=cv2.rectangle(show_frame,(dimp50_viot_pred[0],dimp50_viot_pred[1]),
                                 (dimp50_viot_pred[0]+dimp50_viot_pred[2],dimp50_viot_pred[1]+dimp50_viot_pred[3]),
                                 color_dimp50_viot,thickness=2)
        show_frame=cv2.rectangle(show_frame,(dimp50_rand_pred[0],dimp50_rand_pred[1]),
                                 (dimp50_rand_pred[0]+dimp50_rand_pred[2],dimp50_rand_pred[1]+dimp50_rand_pred[3]),
                                 color_dimp50_rand,thickness=2)

        if writer is None:
            writer = cv2.VideoWriter('../../results/VIOT'+data_name+'_res.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                     (show_frame.shape[1], show_frame.shape[0]))
        writer.write(show_frame)


        


if __name__=='__main__':

    vis_results('../../dataset/VIOT','park_mavic_1')
    vis_results('../../dataset/VIOT','park_mavic_2')
    vis_results('../../dataset/VIOT','park_mavic_3')
    vis_results('../../dataset/VIOT','park_mavic_4')
    vis_results('../../dataset/VIOT','park_mavic_5')
    vis_results('../../dataset/VIOT','park_mavic_6')
    vis_results('../../dataset/VIOT','park_mavic_7')
    vis_results('../../dataset/VIOT','soccerfield_mavic_3')
    vis_results('../../dataset/VIOT','soccerfield_mavic_4')







