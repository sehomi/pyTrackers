import numpy as np
import utm
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy import ndimage


def gps_to_ned(ref_loc, loc):
    y_ref, x_ref, _, _ = utm.from_latlon(ref_loc[0], ref_loc[1])
    pose_ref_utm = np.array( [x_ref, y_ref, -ref_loc[2]] )

    y, x, _, _ = utm.from_latlon(loc[0], loc[1])
    pose_utm = [x,y,-loc[2]]
    pose_ned = pose_utm-pose_ref_utm

    return np.array(pose_ned)

def make_DCM(eul):

    phi = eul[0]
    theta = eul[1]
    psi = eul[2]

    DCM = np.zeros((3,3))
    DCM[0,0] = np.cos(psi)*np.cos(theta)
    DCM[0,1] = np.sin(psi)*np.cos(theta)
    DCM[0,2] = -np.sin(theta)
    DCM[1,0] = np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi)
    DCM[1,1] = np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi)
    DCM[1,2] = np.cos(theta)*np.sin(phi)
    DCM[2,0] = np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi)
    DCM[2,1] = np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi)
    DCM[2,2] = np.cos(theta)*np.cos(phi)

    return DCM
