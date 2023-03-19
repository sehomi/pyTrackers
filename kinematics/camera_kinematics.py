#!/usr/bin/env python

import time
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
from lib.utils_ import plot_kinematics
import matplotlib.pyplot as plt
from kinematics.utils import gps_to_ned, make_DCM

class CameraKinematics:

    def __init__(self, factor, cx, cy, f=None, w=None, h=None, hfov=None, vis=True):

        self._cx = cx
        self._cy = cy
        self._hfov = hfov
        self._w = w
        self._h = h

        if f is not None:
            self._f = f
        elif f is None and (hfov is not None and w is not None and \
                            h is not None):
            self._f = (0.5 * w * (1.0 / np.tan((hfov/2.0)*np.pi/180)));
        else:
            raise ValueError('At least one of arguments "f" or "hfov" must have value.')

        self._init = False

        self._diff = np.array( [0,0,0] )
        self._inertia_dir_before = np.array( [0,0,0] )
        self._inertia_dir_after = np.array( [0,0,0] )
        self._last_rect = (0,0,0,0)
        self._interp_factor = factor
        self._diff_buff = []
        self._pos_buff = []
        self._pos_est = None
        self._pos_buff_size = 40
        self._last_target_states = [False]

        self._vis=vis
        if vis:
            self._fig_3d=plt.figure(0)
            self._ax_3d=plt.axes(projection ='3d')
            self._ax_3d.set_title('Kinematics Plot')
            # self._ax_3d.view_init(elev=-45, azim=45)


    def body_to_inertia(self, body_vec, eul):

        if body_vec is None:
            return None

        ## calculate a DCM and find transpose that takes body to inertial
        DCM_ib = make_DCM(eul).T

        ## return vector in inertial coordinates
        return np.matmul(DCM_ib, body_vec)


    def inertia_to_body(self, in_vec, eul):

        ## calculate a "DCM" using euler angles of camera body, to convert vector
        ## from inertial to body coordinates
        DCM_bi = make_DCM(eul)

        ## return the vector in body coordinates
        return np.matmul(DCM_bi, in_vec)


    def cam_to_body(self, rect):

        if rect is None:
            return None

        ## converting 2d rectangle to a 3d vector in camera coordinates
        vec = self.to_direction_vector(rect, self._cx, self._cy, self._f)

        ## for MAVIC Mini camera, the body axis can be converted to camera
        ## axis by a 90 deg yaw and a 90 deg roll consecutively. then we transpose
        ## it to get camera to body
        DCM_bc = make_DCM([90*np.pi/180, 0, 90*np.pi/180]).T

        return np.matmul(DCM_bc, vec)

    def body_to_cam(self, vec):

        ## for MAVIC Mini camera, the body axis can be converted to camera
        ## axis by a 90 deg yaw and a 90 deg roll consecutively.
        DCM_cb = make_DCM([90*np.pi/180, 0, 90*np.pi/180])

        return np.matmul(DCM_cb, vec)


    def to_direction_vector(self, rect, cx, cy, f):

        ## find center point of target
        center = np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2])

        ## project 2d point from image plane to 3d space using a simple pinhole
        ## camera model
        w = np.array( [ (center[0] - cx) , (center[1] - cy), f] )
        return w/np.linalg.norm(w)

    def from_direction_vector(self, dir, cx, cy, f):

        ## avoid division by zero
        if dir[2] < 0.01:
            dir[2] = 0.01

        ## calculate reprojection of direction vectors to image plane using a
        ## simple pinhole camera model
        X = cx + (dir[0] / dir[2]) * f
        Y = cy + (dir[1] / dir[2]) * f

        return (int(X),int(Y))

    def scale_vector(self, v, z):

        if v is None:
            return None

        ## scale a unit vector v based on the fact that third component should be
        ## equal to z
        max_dist = 50
        if v[2] > 0:
            factor = np.abs(z) / np.abs(v[2])
            if np.linalg.norm(factor*v) < max_dist:
                return factor*v
            else:
                return max_dist*v
        elif v[2] <= 0:
            return max_dist*v

    def limit_vector_to_fov(self, vector):

        ## angle between target direction vector and camera forward axis
        angle = np.arccos( np.dot(vector, np.array([0,0,1])) / np.linalg.norm(vector) )

        ## rotation axis which is perpendicular to both target vector and camera 
        ## axis
        axis = np.cross( np.array([0,0,1]), vector  / np.linalg.norm(vector) )
        last_rotated_vec = None

        for i in range(90):

            rotation_degrees = i * np.sign(angle)
            rotation_radians = np.radians(rotation_degrees)
            rotation_axis = axis / np.linalg.norm(axis)
            rotation_vector = rotation_radians * rotation_axis
            rotation = R.from_rotvec(rotation_vector)
            rotated_vec = rotation.apply( np.array([0,0,1]) )

            ## reproject to image plane
            reproj_vec = self.from_direction_vector(rotated_vec, self._cx, self._cy, self._f)

            if reproj_vec[0] >= self._w or reproj_vec[0] <= 0 or \
               reproj_vec[1] >= self._h or reproj_vec[1] <= 0:

                break

            last_rotated_vec = rotated_vec

        reproj = self.from_direction_vector(vector, self._cx, self._cy, self._f)

        if reproj[0] >= self._w or reproj[0] <= 0 or \
           reproj[1] >= self._h or reproj[1] <= 0:
            # print("out ", reproj_vec)
            return last_rotated_vec
        else:
            # print("in ", reproj)
            return vector
        

    def updateRect3D(self, states, ref, image, rect=None):

        if rect is not None:
            self._last_rect = rect

        ## convert target from a rect in "image coordinates" to a vector
        ## in "camera body coordinates"
        body_dir = self.cam_to_body(rect)

        ## convert target from a vector in "camera body coordinates" to a vector
        ## in "inertial coordinates"
        imu_meas = states[4:7]
        inertia_dir = self.body_to_inertia(body_dir, imu_meas)

        ## convert gps lat, lon positions to a local cartesian coordinate
        ref_loc = ref
        ref_loc[2] = 0
        cam_pos = gps_to_ned(ref_loc, states[1:4])


        if rect is not None:

            ## calculate target pos
            target_pos = self.scale_vector(inertia_dir, cam_pos[2]) + cam_pos

            ## if target is just found, empty the observation buffer to prevent
            ## oscilations around target
            if len(self._last_target_states) >= 5:
                if np.sum( np.array(self._last_target_states[2:5]) ) >= 2  and \
                   np.sum( np.array(self._last_target_states[0:3]) ) <= 1:
                    self._pos_buff = []

            ## buffer target positions
            if len(self._pos_buff) > self._pos_buff_size:
                del self._pos_buff[0]

            self._pos_buff.append([states[0], target_pos[0], target_pos[1], target_pos[2]])

            ## clear buffer after redetection
            # if len(self._pos_buff) > 0:
            #     if states[0] - self._pos_buff[-1][0] > 1.0:
            #         self._pos_buff = []

        ## if target just disappeared, eliminate some of the last buffered observations,
        ## because target's box is having misleading shakes before being lost
        if rect is None and self._last_target_states[-1]:
            for i in range( int(0.2*len(self._pos_buff)) ):
                del self._pos_buff[-1]

        ## record last target states
        if rect is None:
            if len(self._last_target_states) < 5:
                self._last_target_states.append(False)
            else:
                del self._last_target_states[0]
                self._last_target_states.append(False)
        else:
            if len(self._last_target_states) < 5:
                self._last_target_states.append(True)
            else:
                del self._last_target_states[0]
                self._last_target_states.append(True)

        vs = []
        for i in range(1, len(self._pos_buff)):

            t0 = self._pos_buff[i-1][0]
            pos0 = self._pos_buff[i-1][1:4]

            t = self._pos_buff[i][0]
            pos = self._pos_buff[i][1:4]

            dx = np.array(pos) - np.array(pos0)
            dt = t-t0

            if dt < 1 and dt!=0:
                vs.append(dx/dt)

        for data in self._pos_buff:

            pos = data[1:4]
            inertia_dir = pos - cam_pos
            if np.linalg.norm(inertia_dir) == 0:
                continue

            inertia_dir = inertia_dir / np.linalg.norm(inertia_dir)

            ## convert new estimate of target direction vector to body coordinates
            body_dir_est = self.inertia_to_body( inertia_dir, imu_meas)

            ## convert body to cam coordinates
            cam_dir_est = self.body_to_cam(body_dir_est)

            ## reproject to image plane
            center_est = self.from_direction_vector(cam_dir_est, self._cx, self._cy, self._f)

            p1 = (int(center_est[0]-1), int(center_est[1]-1))
            p2 = (int(center_est[0]+1), int(center_est[1]+1))
            image = cv.rectangle(image, p1, p2, (0, 255, 255),2)

        if len(vs)>0:
            v = np.mean(vs,0)
            dt = states[0] - self._pos_buff[-2][0]
            self._pos_est = self._pos_buff[-1][1:4] + self._interp_factor*v*dt

        if self._pos_est is None:
            return self._last_rect

        inertia_dir = self._pos_est - cam_pos
        if np.linalg.norm(inertia_dir) != 0:

            inertia_dir = inertia_dir / np.linalg.norm(inertia_dir)

            ## convert new estimate of target direction vector to body coordinates
            body_dir_est = self.inertia_to_body( inertia_dir, imu_meas)

            ## convert body to cam coordinates
            cam_dir_est = self.body_to_cam(body_dir_est)

            cam_dir_est = self.limit_vector_to_fov(cam_dir_est)

            ## reproject to image plane
            center_est = self.from_direction_vector(cam_dir_est, self._cx, self._cy, self._f)

        ## if target and it's track is lost search image center
        # if len(vs)==0:
        #     center_est = [int(self._cx), int(self._cy)]

        ## estimated rectangle
        rect_est = (int(center_est[0]-self._last_rect[2]/2), \
                    int(center_est[1]-self._last_rect[3]/2),
                    self._last_rect[2], self._last_rect[3])
        # image = cv.putText(image, '{:d}, {:d}, {:d}'.format(center_est[0], center_est[1], len(vs)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 
        #                    1, (0,255,255), 2, cv.LINE_AA)

        return rect_est


    def updateRectSphere(self, imu_meas, rect=None):

        if rect is not None:
            self._last_rect = rect

        ## convert target from a rect in "image coordinates" to a vector
        ## in "camera body coordinates"
        body_dir = self.cam_to_body(rect)

        ## convert target from a vector in "camera body coordinates" to a vector
        ## in "inertial coordinates"
        inertia_dir = self.body_to_inertia(body_dir, imu_meas)

        ## represent inertia_dir in spherecal coordinates
        inertia_dir_sp = self.toSpherecalCoords(inertia_dir)

        if self._init:
            diff=np.array([0.0, 0.0, 0.0])
            if inertia_dir is not None:
                ## find the difference between new observation (inertia_dir) and last
                ## known direction (self._inertia_dir_before)
                diff = np.array([0.0, self.angleDifference(inertia_dir_sp[1], self._inertia_dir_before[1]), \
                                      self.angleDifference(inertia_dir_sp[2], self._inertia_dir_before[2])])


            ## if target is just found, empty the observation buffer to prevent
            ## oscilations around target
            if inertia_dir is not None and all(~np.array(self._last_target_states)):
                self._diff_buff = []

            ## make the differences smooth overtime by a moving average. this adds a dynamic to target
            ## direction vector.
            if len(self._diff_buff) > -self._interp_factor:
                del self._diff_buff[0]
                self._diff_buff.append(diff)
            else:
                self._diff_buff.append(diff)

            ## if target just disappeared, eliminate some of the last buffered observations,
            ## because target's box is having misleading shakes before being lost
            if inertia_dir is None and self._last_target_states[-1]:
                for i in range( int(0.4*len(self._diff_buff)) ):
                    del self._diff_buff[-1]

            ## record last target states
            if inertia_dir is None:
                if len(self._last_target_states) < 3:
                    self._last_target_states.append(False)
                else:
                    del self._last_target_states[0]
                    self._last_target_states.append(False)
            else:
                if len(self._last_target_states) < 3:
                    self._last_target_states.append(True)
                else:
                    del self._last_target_states[0]
                    self._last_target_states.append(True)

            self._diff = np.mean(self._diff_buff, 0)

            ## calculate new estimate for target's direction vector
            self._inertia_dir_after = self._inertia_dir_before + self._diff

            ## save this new estimate as last known direction in the memory
            self._inertia_dir_before = self._inertia_dir_after.copy()

        else:

            if inertia_dir is not None:
                ## initialize with first observation
                self._inertia_dir_before = inertia_dir_sp.copy()
                self._inertia_dir_after = inertia_dir_sp.copy()
                self._diff = np.array([0.0,0.0,0.0])

                self._init = True
            else:
                return None

        ## convert back to cartesian coordinates
        inertia_dir_after_ca = self.toCartesianCoords(self._inertia_dir_after)

        if self._vis:
            ## expressing camera frame by for vectors of its image corners in inertial
            ## frame
            corners = self.get_camera_frame_vecs(imu_meas,self._w,self._h)
            plot_kinematics(imu_meas, inertia_dir_after_ca, self._ax_3d, corners)

        ## convert new estimate of target direction vector to body coordinates
        body_dir_est = self.inertia_to_body( inertia_dir_after_ca, imu_meas)

        ## convert body to cam coordinates
        cam_dir_est = self.body_to_cam(body_dir_est)

        ## reproject to image plane
        center_est = self.from_direction_vector(cam_dir_est, self._cx, self._cy, self._f)

        ## estimated rectangle
        rect_est = (int(center_est[0]-self._last_rect[2]/2), \
                    int(center_est[1]-self._last_rect[3]/2),
                    self._last_rect[2], self._last_rect[3])

        return rect_est

    def updateRect(self, imu_meas, rect=None):

        if rect is not None:
            self._last_rect = rect

        ## convert target from a rect in "image coordinates" to a vector
        ## in "camera body coordinates"
        body_dir = self.cam_to_body(rect)

        ## convert target from a vector in "camera body coordinates" to a vector
        ## in "inertial coordinates"
        inertia_dir = self.body_to_inertia(body_dir, imu_meas)

        if self._init:
            ## update difference vector only in case of new observation
            ## otherwise continue changing direction vector with last know
            ## speed
            diff=np.array([0.0,0.0,0.0])
            if inertia_dir is not None:
                ## find the difference between new observation (inertia_dir) and last
                ## known direction (self._inertia_dir_before)
                diff = inertia_dir - self._inertia_dir_before

                ## make the differences smooth overtime. this add a dynamic to target
                ## direction vector.
                self._diff = self._interp_factor*self._diff + (1-self._interp_factor)*diff

            ## calculate new estimate for target's direction vector
            self._inertia_dir_after = self._inertia_dir_before + self._diff

            ## ensure direction vector always has a length of 1
            self._inertia_dir_after = self._inertia_dir_after/np.linalg.norm(self._inertia_dir_after)

            ## save this new estimate as last known direction in the memory
            self._inertia_dir_before = self._inertia_dir_after.copy()

        else:

            if inertia_dir is not None:
                ## initialize with first observation
                self._inertia_dir_before = inertia_dir
                self._inertia_dir_after = inertia_dir
                self._diff = np.array([0.0,0.0,0.0])

                self._init = True
            else:
                return None

        if self._vis:
            ## expressing camera frame by for vectors of its image corners in inertial
            ## frame
            corners = self.get_camera_frame_vecs(imu_meas,self._w,self._h)
            plot_kinematics(imu_meas,self._inertia_dir_after,self._ax_3d,corners)

        ## convert new estimate of target direction vector to body coordinates
        body_dir_est = self.inertia_to_body(self._inertia_dir_after,imu_meas)

        ## convert body to cam coordinates
        cam_dir_est = self.body_to_cam(body_dir_est)

        ## reproject to image plane
        center_est = self.from_direction_vector(cam_dir_est, self._cx, self._cy, self._f)

        ## estimated rectangle
        rect_est = (int(center_est[0]-self._last_rect[2]/2), \
                    int(center_est[1]-self._last_rect[3]/2),
                    self._last_rect[2], self._last_rect[3])

        return rect_est


    def get_camera_frame_vecs(self, eul, w, h):

        ## convert image corners from a point in "image coordinates" to a vector
        ## in "camera body coordinates"
        top_left = self.cam_to_body([-1,-1,2,2])
        top_right = self.cam_to_body([w-1,-1,2,2])
        bottom_left = self.cam_to_body([-1,h-1,2,2])
        bottom_right = self.cam_to_body([w-1,h-1,2,2])

        ## convert image corners from a vector in "camera body coordinates" to
        ## a vector in "inertial coordinates"
        top_left_inertia_dir = self.body_to_inertia(top_left, eul)
        top_right_inertia_dir = self.body_to_inertia(top_right, eul)
        bottom_left_inertia_dir = self.body_to_inertia(bottom_left, eul)
        bottom_right_inertia_dir = self.body_to_inertia(bottom_right, eul)


        return (top_left_inertia_dir,top_right_inertia_dir,\
                bottom_left_inertia_dir,bottom_right_inertia_dir)


    def toSpherecalCoords(self, vec):

        if vec is None:
            return None

        x = vec[0]
        y = vec[1]
        z = vec[2]

        r = np.sqrt(x**2 + y**2 + z**2)
        th = np.arccos( z / r )

        if x>0:
            phi = np.arctan(y/x)
        elif x<0 and y>=0:
            phi = np.arctan(y/x) + np.pi
        elif x<0 and y<0:
            phi = np.arctan(y/x) - np.pi
        elif x==0 and y>0:
            phi = np.pi
        elif x==0 and y<0:
            phi = -np.pi

        return np.array([r,th, phi])

    def toCartesianCoords(self, vec):

        if vec is None:
            return None

        r = vec[0]
        th = vec[1]
        phi = vec[2]

        x = r*np.cos(phi)*np.sin(th)
        y = r*np.sin(phi)*np.sin(th)
        z = r*np.cos(th)

        return np.array([x, y, z])

    def angleDifference(self, ang1, ang2):
        PI  = np.pi

        a = ang1 - ang2
        if a > PI:
            a -= 2*PI
        if a < -PI:
            a += 2*PI

        return a
