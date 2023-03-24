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
from lib.utils_ import get_img_list
from examples.vis.draw_plot import get_preds_by_name
from lib.utils_ import get_thresh_success_pair,calAUC
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
    f = open('../all_results_mixformer.json', 'r')
    results = json.load(f)
    if not data_name in results.keys():
        raise ValueError
    # cap=cv2.VideoCapture(data_name+'_vis.avi')
    data_all = results[data_name]

    gts = get_preds_by_name(data_all, 'gts')
    kcf_hog_preds = get_preds_by_name(data_all, 'kcf_hog_preds')
    ldes_preds = get_preds_by_name(data_all, 'ldes_preds')
    strcf_preds = get_preds_by_name(data_all, 'strcf_preds')
    csrdcf_preds = get_preds_by_name(data_all, 'csrdcf_preds')
    dimp50_preds = get_preds_by_name(data_all, 'tracker_dimp50_preds')
    prdimp50_preds = get_preds_by_name(data_all, 'tracker_prdimp50_preds')
    tomp_preds = get_preds_by_name(data_all, 'tracker_tomp_preds')
    mixformer_preds = get_preds_by_name(data_all, 'tracker_mixformer_preds')
    kys_preds = get_preds_by_name(data_all, 'tracker_kys_preds')

    img_dir=os.path.join(dataset_dir,data_name)

    img_list=get_img_list(img_dir)
    start_frame,end_frame=dataset_config.frames[data_name][0:2]
    img_list=img_list[start_frame-1:end_frame]
    imgs = []

    color_gt=(0,0,0) # gt
    color_ldes=(240,32,160) # 
    color_strcf=(32,240,160) # 
    color_csrdcf=(100,0,100) #
    color_kcf=(160,32,240) #
    color_dimp50=(0,255,0) #
    color_prdimp50=(0,0,255) #
    color_kys=(255,0,0) #
    color_tomp=(0,255,255) #
    color_mixformer=(0,165,255) #

    writer=None
    for i in range(len(img_list)):

        # flag,current_frame=cap.read()
        # if flag is False:
        #     return

        current_frame = cv2.imread(img_list[i])
        gt=gts[i]
        kcf_hog_pred = kcf_hog_preds[i]
        ldes_pred = ldes_preds[i]
        strcf_pred = strcf_preds[i]
        csrdcf_pred = csrdcf_preds[i]
        dimp50_pred = dimp50_preds[i]
        prdimp50_pred = prdimp50_preds[i]
        tomp_pred = tomp_preds[i]
        mixformer_pred = mixformer_preds[i]
        kys_pred = kys_preds[i]
        
        show_frame=drawrect(current_frame,(kcf_hog_pred[0],kcf_hog_pred[1]),
                                 (kcf_hog_pred[0]+kcf_hog_pred[2],kcf_hog_pred[1]+kcf_hog_pred[3]),
                                 color_kcf,thickness=2)
        show_frame=drawrect(show_frame,(ldes_pred[0],ldes_pred[1]),
                                 (ldes_pred[0]+ldes_pred[2],ldes_pred[1]+ldes_pred[3]),
                                 color_ldes,thickness=2)
        show_frame=drawrect(show_frame,(strcf_pred[0],strcf_pred[1]),
                                 (strcf_pred[0]+strcf_pred[2],strcf_pred[1]+strcf_pred[3]),
                                 color_strcf,thickness=2)
        show_frame=drawrect(show_frame,(csrdcf_pred[0],csrdcf_pred[1]),
                                 (csrdcf_pred[0]+csrdcf_pred[2],csrdcf_pred[1]+csrdcf_pred[3]),
                                 color_csrdcf,thickness=2)
        show_frame=drawrect(show_frame,(dimp50_pred[0],dimp50_pred[1]),
                                 (dimp50_pred[0]+dimp50_pred[2],dimp50_pred[1]+dimp50_pred[3]),
                                 color_dimp50,thickness=2)
        show_frame=drawrect(show_frame,(prdimp50_pred[0],prdimp50_pred[1]),
                                 (prdimp50_pred[0]+prdimp50_pred[2],prdimp50_pred[1]+prdimp50_pred[3]),
                                 color_prdimp50,thickness=2)
        show_frame=drawrect(show_frame,(kys_pred[0],kys_pred[1]),
                                 (kys_pred[0]+kys_pred[2],kys_pred[1]+kys_pred[3]),
                                 color_kys,thickness=2)
        show_frame=drawrect(show_frame,(tomp_pred[0],tomp_pred[1]),
                                 (tomp_pred[0]+tomp_pred[2],tomp_pred[1]+tomp_pred[3]),
                                 color_tomp,thickness=2)
        show_frame=drawrect(show_frame,(mixformer_pred[0],mixformer_pred[1]),
                                 (mixformer_pred[0]+mixformer_pred[2],mixformer_pred[1]+mixformer_pred[3]),
                                 color_mixformer,thickness=2)

        # threshes,kcf_success=get_thresh_success_pair(gts,kcf_hog_preds)
        # _,csk_success=get_thresh_success_pair(gts,csk_preds)
        # _,mosse_success=get_thresh_success_pair(gts,mosse_preds)
        # _,dsst_success=get_thresh_success_pair(gts,dsst_preds)
        # _,ecohc_success=get_thresh_success_pair(gts,eco_hc_preds)
        # _,gts_success=get_thresh_success_pair(gts,gts)
        # show_frame = cv2.putText(show_frame, 'MOSSE ' + str(calAUC(mosse_success))[:5], (50, 40),
        #                             cv2.FONT_HERSHEY_COMPLEX, 1, color_mosse)
        # show_frame=cv2.putText(show_frame,'KCF '+str(calAUC(kcf_success))[:5],(50,120),cv2.FONT_HERSHEY_COMPLEX,1,color_kcf)
        # show_frame=cv2.putText(show_frame,'CSK '+str(calAUC(csk_success))[:5],(50,80),cv2.FONT_HERSHEY_COMPLEX,1,color_csk)
        # show_frame=cv2.putText(show_frame,'ECO '+str(calAUC(dsst_success))[:5],(50,200),cv2.FONT_HERSHEY_COMPLEX,1,color_eco)
        # show_frame=cv2.putText(show_frame,'DSST '+str(calAUC(ecohc_success))[:5],(50,160),cv2.FONT_HERSHEY_COMPLEX,1,color_dsst)
        # show_frame=cv2.putText(show_frame,'GT',(50,240),cv2.FONT_HERSHEY_COMPLEX,1,color_gt)
        
        # cv2.imshow('demo',show_frame)
        # cv2.waitKey(10)

        imgs.append(show_frame)

    f = open('../all_results_mixformer_viot.json', 'r')
    results = json.load(f)
    if not data_name in results.keys():
        raise ValueError
    # cap=cv2.VideoCapture(data_name+'_vis.avi')
    data_all = results[data_name]

    gts = get_preds_by_name(data_all, 'gts')
    kcf_hog_preds = get_preds_by_name(data_all, 'kcf_hog_preds')
    ldes_preds = get_preds_by_name(data_all, 'ldes_preds')
    strcf_preds = get_preds_by_name(data_all, 'strcf_preds')
    csrdcf_preds = get_preds_by_name(data_all, 'csrdcf_preds')
    dimp50_preds = get_preds_by_name(data_all, 'tracker_dimp50_preds')
    prdimp50_preds = get_preds_by_name(data_all, 'tracker_prdimp50_preds')
    tomp_preds = get_preds_by_name(data_all, 'tracker_tomp_preds')
    mixformer_preds = get_preds_by_name(data_all, 'tracker_mixformer_preds')
    kys_preds = get_preds_by_name(data_all, 'tracker_kys_preds')

    writer=None
    for i in range(len(imgs)):

        # flag,current_frame=cap.read()
        # if flag is False:
        #     return

        current_frame = imgs[i]
        gt=gts[i]
        kcf_hog_pred = kcf_hog_preds[i]
        ldes_pred = ldes_preds[i]
        strcf_pred = strcf_preds[i]
        csrdcf_pred = csrdcf_preds[i]
        dimp50_pred = dimp50_preds[i]
        prdimp50_pred = prdimp50_preds[i]
        tomp_pred = tomp_preds[i]
        mixformer_pred = mixformer_preds[i]
        kys_pred = kys_preds[i]

        try:
            show_frame=cv2.rectangle(current_frame,(gt[0],gt[1]),
                                    (gt[0]+gt[2],gt[1]+gt[3]),
                                    color_gt,thickness=2)
        except:
            pass

        show_frame=cv2.rectangle(show_frame,(kcf_hog_pred[0],kcf_hog_pred[1]),
                                 (kcf_hog_pred[0]+kcf_hog_pred[2],kcf_hog_pred[1]+kcf_hog_pred[3]),
                                 color_kcf,thickness=2)
        show_frame=cv2.rectangle(show_frame,(ldes_pred[0],ldes_pred[1]),
                                 (ldes_pred[0]+ldes_pred[2],ldes_pred[1]+ldes_pred[3]),
                                 color_ldes,thickness=2)
        show_frame=cv2.rectangle(show_frame,(strcf_pred[0],strcf_pred[1]),
                                 (strcf_pred[0]+strcf_pred[2],strcf_pred[1]+strcf_pred[3]),
                                 color_strcf,thickness=2)
        show_frame=cv2.rectangle(show_frame,(csrdcf_pred[0],csrdcf_pred[1]),
                                 (csrdcf_pred[0]+csrdcf_pred[2],csrdcf_pred[1]+csrdcf_pred[3]),
                                 color_csrdcf,thickness=2)
        show_frame=cv2.rectangle(show_frame,(dimp50_pred[0],dimp50_pred[1]),
                                 (dimp50_pred[0]+dimp50_pred[2],dimp50_pred[1]+dimp50_pred[3]),
                                 color_dimp50,thickness=2)
        show_frame=cv2.rectangle(show_frame,(prdimp50_pred[0],prdimp50_pred[1]),
                                 (prdimp50_pred[0]+prdimp50_pred[2],prdimp50_pred[1]+prdimp50_pred[3]),
                                 color_prdimp50,thickness=2)
        show_frame=cv2.rectangle(show_frame,(kys_pred[0],kys_pred[1]),
                                 (kys_pred[0]+kys_pred[2],kys_pred[1]+kys_pred[3]),
                                 color_kys,thickness=2)
        show_frame=cv2.rectangle(show_frame,(tomp_pred[0],tomp_pred[1]),
                                 (tomp_pred[0]+tomp_pred[2],tomp_pred[1]+tomp_pred[3]),
                                 color_tomp,thickness=2)
        show_frame=cv2.rectangle(show_frame,(mixformer_pred[0],mixformer_pred[1]),
                                 (mixformer_pred[0]+mixformer_pred[2],mixformer_pred[1]+mixformer_pred[3]),
                                 color_mixformer,thickness=2)

        # threshes,kcf_success=get_thresh_success_pair(gts,kcf_hog_preds)
        # _,csk_success=get_thresh_success_pair(gts,csk_preds)
        # _,mosse_success=get_thresh_success_pair(gts,mosse_preds)
        # _,dsst_success=get_thresh_success_pair(gts,dsst_preds)
        # _,ecohc_success=get_thresh_success_pair(gts,eco_hc_preds)
        # _,gts_success=get_thresh_success_pair(gts,gts)
        # show_frame = cv2.putText(show_frame, 'MOSSE ' + str(calAUC(mosse_success))[:5], (50, 40),
        #                             cv2.FONT_HERSHEY_COMPLEX, 1, color_mosse)
        # show_frame=cv2.putText(show_frame,'KCF '+str(calAUC(kcf_success))[:5],(50,120),cv2.FONT_HERSHEY_COMPLEX,1,color_kcf)
        # show_frame=cv2.putText(show_frame,'CSK '+str(calAUC(csk_success))[:5],(50,80),cv2.FONT_HERSHEY_COMPLEX,1,color_csk)
        # show_frame=cv2.putText(show_frame,'ECO '+str(calAUC(dsst_success))[:5],(50,200),cv2.FONT_HERSHEY_COMPLEX,1,color_eco)
        # show_frame=cv2.putText(show_frame,'DSST '+str(calAUC(ecohc_success))[:5],(50,160),cv2.FONT_HERSHEY_COMPLEX,1,color_dsst)
        # show_frame=cv2.putText(show_frame,'GT',(50,240),cv2.FONT_HERSHEY_COMPLEX,1,color_gt)
        
        # cv2.imshow('demo',show_frame)
        # cv2.waitKey(10)


        if writer is None:
            writer = cv2.VideoWriter('../../results/VIOT/'+data_name+'_res.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                     (show_frame.shape[1], show_frame.shape[0]))
        writer.write(show_frame)
        


if __name__=='__main__':

    vis_results('../../dataset/VIOT','soccerfield_mavic_3')







