import cv2
import numpy as np
import importlib
import os
from collections import OrderedDict
from lib.utils_ import get_img_list,get_states_data,get_ground_truthes,get_ground_truthes_viot,APCE,PSR
# from cftracker.mosse import MOSSE
# from cftracker.csk import CSK
# from cftracker.kcf import KCF
# from cftracker.cn import CN
# from cftracker.dsst import DSST
# from cftracker.staple import Staple
# from cftracker.dat import DAT
# from cftracker.eco import ECO
# from cftracker.bacf import BACF
# from cftracker.csrdcf import CSRDCF
# from cftracker.samf import SAMF
# from cftracker.ldes import LDES
# from cftracker.mkcfup import MKCFup
# from cftracker.strcf import STRCF
# from cftracker.mccth_staple import MCCTHStaple
# from lib.eco.config import otb_deep_config,otb_hc_config
# from cftracker.config import staple_config,ldes_config,dsst_config,csrdcf_config,mkcf_up_config,mccth_staple_config

from kinematics.camera_kinematics import CameraKinematics

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

class PyTracker:
    def __init__(self,img_dir,tracker_type,dataset_config):
        self.img_dir=img_dir
        self.tracker_type=tracker_type
        self.frame_list = get_img_list(img_dir)
        self.frame_list.sort()
        # dataname=img_dir.split('/')[-2]
        dataname=img_dir.split('/')[-1] ## VIOT
        # self.gts=get_ground_truthes(img_dir[:-4])
        self.gts=get_ground_truthes_viot(img_dir) ## VIOT
        self.states=get_states_data(img_dir) ## VIOT
        self.fov=dataset_config.fov[dataname]
        self.ethTracker=False

        if dataname in dataset_config.frames.keys():
            start_frame,end_frame=dataset_config.frames[dataname][0:2]
            self.init_gt=self.gts[start_frame-1]

            self.frame_list=self.frame_list[start_frame-1:end_frame]
            self.states=self.states[start_frame-1:end_frame]

        else:
            self.init_gt=self.gts[0]
        if self.tracker_type == 'MOSSE':
            self.tracker=MOSSE()
            self.ratio_thresh=0.1
        elif self.tracker_type=='CSK':
            self.tracker=CSK()
            self.ratio_thresh=0.1
        elif self.tracker_type=='CN':
            self.tracker=CN()
            self.ratio_thresh=0.1
        elif self.tracker_type=='DSST':
            self.tracker=DSST(dsst_config.DSSTConfig())
            self.ratio_thresh=0.1
        elif self.tracker_type=='Staple':
            self.tracker=Staple(config=staple_config.StapleConfig())
            self.ratio_thresh=0.1
        elif self.tracker_type=='Staple-CA':
            self.tracker=Staple(config=staple_config.StapleCAConfig())
            self.ratio_thresh=0.1
        elif self.tracker_type=='KCF_CN':
            self.tracker=KCF(features='cn',kernel='gaussian')
            self.ratio_thresh=0.8
        elif self.tracker_type=='KCF_GRAY':
            self.tracker=KCF(features='gray',kernel='gaussian')
            self.ratio_thresh=0.8
        elif self.tracker_type=='KCF_HOG':
            self.tracker=KCF(features='hog',kernel='gaussian')
            try:
                self.ratio_thresh=dataset_config.params['KCF_HOG'][dataname][0]
            except:
                self.ratio_thresh=0.1

            try:
                self.interp_factor=dataset_config.params['KCF_HOG'][dataname][1]
            except:
                self.interp_factor=0.3

        elif self.tracker_type=='DCF_GRAY':
            self.tracker=KCF(features='gray',kernel='linear')
            self.ratio_thresh=0.1
        elif self.tracker_type=='DCF_HOG':
            self.tracker=KCF(features='hog',kernel='linear')
            self.ratio_thresh=0.1
        elif self.tracker_type=='DAT':
            self.tracker=DAT()
            self.ratio_thresh=0.1
        elif self.tracker_type=='ECO-HC':
            self.tracker=ECO(config=otb_hc_config.OTBHCConfig())
            self.ratio_thresh=0.5
        elif self.tracker_type=='ECO':
            self.tracker=ECO(config=otb_deep_config.OTBDeepConfig())
            try:
                self.ratio_thresh=dataset_config.params['ECO'][dataname][0]
            except:
                self.ratio_thresh=0.5

            try:
                self.interp_factor=dataset_config.params['ECO'][dataname][1]
            except:
                self.interp_factor=0.3

        elif self.tracker_type=='BACF':
            self.tracker=BACF()
            self.ratio_thresh=0.2
        elif self.tracker_type=='CSRDCF':
            self.tracker=CSRDCF(config=csrdcf_config.CSRDCFConfig())
            try:
                self.ratio_thresh=dataset_config.params['CSRDCF'][dataname][0]
            except:
                self.ratio_thresh=0.3

            try:
                self.interp_factor=dataset_config.params['CSRDCF'][dataname][1]
            except:
                self.interp_factor=0.3

        elif self.tracker_type=='CSRDCF-LP':
            self.tracker=CSRDCF(config=csrdcf_config.CSRDCFLPConfig())
            self.ratio_thresh=0.1
        elif self.tracker_type=='SAMF':
            self.tracker=SAMF()
            self.ratio_thresh=0.1
        elif self.tracker_type=='LDES':
            self.tracker=LDES(ldes_config.LDESDemoLinearConfig())
            try:
                self.ratio_thresh=dataset_config.params['LDES'][dataname][0]
            except:
                self.ratio_thresh=0.1

            try:
                self.interp_factor=dataset_config.params['LDES'][dataname][1]
            except:
                self.interp_factor=0.3

        elif self.tracker_type=='DSST-LP':
            self.tracker=DSST(dsst_config.DSSTLPConfig())
            self.ratio_thresh=0.1
        elif self.tracker_type=='MKCFup':
            self.tracker=MKCFup(config=mkcf_up_config.MKCFupConfig())
            self.ratio_thresh=0.1
        elif self.tracker_type=='MKCFup-LP':
            self.tracker=MKCFup(config=mkcf_up_config.MKCFupLPConfig())
            self.ratio_thresh=0.1
        elif self.tracker_type=='STRCF':
            self.tracker=STRCF()
            try:
                self.ratio_thresh=dataset_config.params['STRCF'][dataname][0]
            except:
                self.ratio_thresh=0.25

            try:
                self.interp_factor=dataset_config.params['STRCF'][dataname][1]
            except:
                self.interp_factor=0.3

        elif self.tracker_type=='MCCTH-Staple':
            self.tracker=MCCTHStaple(config=mccth_staple_config.MCCTHOTBConfig())
            self.ratio_thresh=0.1

        elif self.tracker_type=='MCCTH':
            self.tracker=MCCTH(config=mccth_config.MCCTHConfig())
            self.ratio_thresh=0.1

        elif self.tracker_type=='DIMP50':
            self.tracker=self.getETHTracker('dimp','dimp50')
            self.ethTracker=True
            try:
                self.ratio_thresh=dataset_config.params['DIMP50'][dataname][0]
            except:
                self.ratio_thresh=0.5

            try:
                self.interp_factor=dataset_config.params['DIMP50'][dataname][1]
            except:
                self.interp_factor=0.3

        elif self.tracker_type=='PRDIMP50':
            self.tracker=self.getETHTracker('dimp','prdimp50')
            self.ethTracker=True
            try:
                self.ratio_thresh=dataset_config.params['PRDIMP50'][dataname][0]
            except:
                self.ratio_thresh=0.5

            try:
                self.interp_factor=dataset_config.params['PRDIMP50'][dataname][1]
            except:
                self.interp_factor=0.3

        elif self.tracker_type=='KYS':
            self.tracker=self.getETHTracker('kys','default')
            self.ethTracker=True
            try:
                self.ratio_thresh=dataset_config.params['KYS'][dataname][0]
            except:
                self.ratio_thresh=0.5

            try:
                self.interp_factor=dataset_config.params['KYS'][dataname][1]
            except:
                self.interp_factor=0.3

        elif self.tracker_type=='TOMP':
            self.tracker=self.getETHTracker('tomp','tomp101')
            self.ethTracker=True
            try:
                self.ratio_thresh=dataset_config.params['TOMP'][dataname][0]
            except:
                self.ratio_thresh=0.5

            try:
                self.interp_factor=dataset_config.params['TOMP'][dataname][1]
            except:
                self.interp_factor=0.3

        elif self.tracker_type=='MIXFORMER_VIT':
            from lib.test.tracker.mixformer_vit_online import MixFormerOnline
            from lib.test.parameter.mixformer_vit_online import parameters

            params = parameters('baseline', 'mixformer_vit_base_online.pth.tar', 5.05)
            self.tracker = MixFormerOnline(params, 'got10k_test')
            
        else:
            raise NotImplementedError

        self.viot = True
        # self.viot = False


    def getETHTracker(self, name, params):
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(name, params))
        params = param_module.parameters()
        params.tracker_name = name
        params.param_name = params

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tracker', name))
        tracker_module = importlib.import_module('pytracking.tracker.{}'.format(name))
        tracker_class = tracker_module.get_tracker_class()

        tracker = tracker_class(params)

        if hasattr(tracker, 'initialize_features'):
            tracker.initialize_features()

        return tracker

    def initETHTracker(self, frame, bbox):

        x, y, w, h = bbox
        init_state = [x, y, w, h]
        box = {'init_bbox': init_state, 'init_object_ids': [1, ], 'object_ids': [1, ],
                    'sequence_object_ids': [1, ]}
        self.tracker.initialize(frame, box)

    def doTrack(self, current_frame, verbose, est_loc, do_learning, viot=False):
    	if self.ethTracker:
            if viot:
                out = self.tracker.track(current_frame, est_loc, do_learning=do_learning)
            else:
        	    out = self.tracker.track(current_frame)

            bbox = [int(s) for s in out['target_bbox']]
    	else:
    	    if viot:
    	        bbox=self.tracker.update(current_frame,vis=verbose,FI=est_loc, \
    	                                 do_learning=do_learning) ## VIOT
    	    else:
    	    	bbox=self.tracker.update(current_frame,vis=verbose)
    	    	# bbox=self.tracker.update(current_frame,vis=verbose,FI=est_loc)

    	return bbox


    def tracking(self,verbose=True,video_path=None):
        poses = []
        ratios = []
        init_frame = cv2.imread(self.frame_list[0])
        #print(init_frame.shape)
        init_gt = np.array(self.init_gt)
        x1, y1, w, h =init_gt
        init_gt=tuple(init_gt)
        if self.ethTracker:
            self.initETHTracker(init_frame, init_gt)
        else:
            self.tracker.init(init_frame,init_gt)
        writer=None
        if verbose is True and video_path is not None:
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (init_frame.shape[1], init_frame.shape[0]))
            ratios_path = os.path.splitext(video_path)[0] + ".txt"

        ## kinematic model for MAVIC Mini with horizontal field of view (hfov)
        ## equal to 66 deg.
        kin = CameraKinematics(self.interp_factor, init_frame.shape[1]/2, init_frame.shape[0]/2,\
                                w=init_frame.shape[1], h=init_frame.shape[0],\
                                hfov=self.fov, vis=False)

        psr0=-1
        psr=-1
        est_loc=init_gt
        stop=False
        last_bbox=None

        for idx in range(len(self.frame_list)):
            if idx != 0:
                current_frame=cv2.imread(self.frame_list[idx])
                height,width=current_frame.shape[:2]

                if stop:
                    bbox=last_bbox
                else:
                    bbox=self.doTrack(current_frame, verbose, est_loc, psr/psr0>self.ratio_thresh and not stop, viot=self.viot)
                    last_bbox=bbox

                stop=bbox[2] > width or bbox[3] > height

                ## evaluating tracked target
                apce = APCE(self.tracker.score)
                if self.ethTracker:
                    psr = apce
                else:
                    psr = PSR(self.tracker.score)
                F_max = np.max(self.tracker.score)

                if psr0 is -1: psr0=psr

                ratios.append(psr/psr0)

                ## estimating target location using kinematc model
                if psr/psr0 > self.ratio_thresh:
                    est_loc = kin.updateRect3D(self.states[idx,:], self.states[0,1:4], current_frame, bbox)
                else:
                    est_loc = kin.updateRect3D(self.states[idx,:], self.states[0,1:4], current_frame, None)


                # print("psr ratio: ",psr/psr0, " learning: ", psr/psr0 > self.ratio_thresh, " est: ", est_loc)

                x1,y1,w,h=bbox
                if verbose is True:
                    if len(current_frame.shape)==2:
                        current_frame=cv2.cvtColor(current_frame,cv2.COLOR_GRAY2BGR)
                    score = self.tracker.score
                    # apce = APCE(score)
                    # psr = PSR(score)
                    # F_max = np.max(score)
                    size=self.tracker.crop_size
                    score = cv2.resize(score, size)
                    score -= score.min()
                    score =score/ score.max()
                    score = (score * 255).astype(np.uint8)
                    # score = 255 - score
                    score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
                    center = (int(x1+w/2-self.tracker.trans[1]),int(y1+h/2-self.tracker.trans[0]))
                    x0,y0=center
                    x0=np.clip(x0,0,width-1)
                    y0=np.clip(y0,0,height-1)
                    center=(x0,y0)
                    xmin = int(center[0]) - size[0] // 2
                    xmax = int(center[0]) + size[0] // 2 + size[0] % 2
                    ymin = int(center[1]) - size[1] // 2
                    ymax = int(center[1]) + size[1] // 2 + size[1] % 2
                    left = abs(xmin) if xmin < 0 else 0
                    xmin = 0 if xmin < 0 else xmin
                    right = width - xmax
                    xmax = width if right < 0 else xmax
                    right = size[0] + right if right < 0 else size[0]
                    top = abs(ymin) if ymin < 0 else 0
                    ymin = 0 if ymin < 0 else ymin
                    down = height - ymax
                    ymax = height if down < 0 else ymax
                    down = size[1] + down if down < 0 else size[1]
                    score = score[top:down, left:right]
                    crop_img = current_frame[ymin:ymax, xmin:xmax]
                    score_map = cv2.addWeighted(crop_img, 0.6, score, 0.4, 0)
                    current_frame[ymin:ymax, xmin:xmax] = score_map
                    show_frame=cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0),2)

                    if self.tracker_type=='DIMP50' or self.tracker_type=='KYS' or self.tracker_type=='TOMP' or self.tracker_type=='PRDIMP50':
                        for zone in self.tracker._sample_coords:
                            show_frame=cv2.rectangle(show_frame, (int(zone[1]), int(zone[0])), 
                                                     (int(zone[3]), int(zone[2])), (0, 255, 255),1)

                    if not psr/psr0>self.ratio_thresh:
                        show_frame = cv2.line(show_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 2)
                        show_frame = cv2.line(show_frame, (int(x1+w), int(y1)), (int(x1), int(y1 + h)), (0, 0, 255), 2)

                    if self.viot:
                        p1 = (int(est_loc[0]+est_loc[2]/2-1), int(est_loc[1]+est_loc[3]/2-1))
                        p2 = (int(est_loc[0]+est_loc[2]/2+1), int(est_loc[1]+est_loc[3]/2+1))
                        show_frame = cv2.rectangle(show_frame, p1, p2, (255, 0, 0),2)

                    # cv2.putText(show_frame, 'APCE:' + str(apce)[:5], (0, 250), cv2.FONT_HERSHEY_COMPLEX, 2,
                    #             (0, 0, 255), 5)
                    # cv2.putText(show_frame, 'PSR:' + str(psr)[:5], (0, 300), cv2.FONT_HERSHEY_COMPLEX, 2,
                    #             (255, 0, 0), 5)
                    # cv2.putText(show_frame, 'Fmax:' + str(F_max)[:5], (0, 350), cv2.FONT_HERSHEY_COMPLEX, 2,
                    #             (255, 0, 0), 5)

                    if not IN_COLAB:
                        cv2.imshow('demo', show_frame)
                        
                    if writer is not None:
                        writer.write(show_frame)
                    cv2.waitKey(1)

            poses.append(np.array([int(x1), int(y1), int(w), int(h)]))

        np.savetxt(ratios_path, np.array(ratios), delimiter=',')
        return np.array(poses)
