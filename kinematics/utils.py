import numpy as np
import utm
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy import ndimage
from mpl_toolkits.mplot3d import proj3d

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


def make_get_proj(self, rx, ry, rz):
    '''
    Return a variation on :func:`~mpl_toolkit.mplot2d.axes3d.Axes3D.getproj` that
    makes the box aspect ratio equal to *rx:ry:rz*, using an axes object *self*.
    '''

    rm = max(rx, ry, rz)
    kx = rm / rx; ky = rm / ry; kz = rm / rz;

    # Copied directly from mpl_toolkit/mplot3d/axes3d.py. New or modified lines are
    # marked by ##
    def get_proj():
        relev, razim = np.pi * self.elev/180, np.pi * self.azim/180

        xmin, xmax = self.get_xlim3d()
        ymin, ymax = self.get_ylim3d()
        zmin, zmax = self.get_zlim3d()

        # transform to uniform world coordinates 0-1.0,0-1.0,0-1.0
        worldM = proj3d.world_transformation(xmin, xmax,
                                             ymin, ymax,
                                             zmin, zmax)

        # adjust the aspect ratio                          ##
        aspectM = proj3d.world_transformation(-kx + 1, kx, ##
                                              -ky + 1, ky, ##
                                              -kz + 1, kz) ##

        # look into the middle of the new coordinates
        R = np.array([0.5, 0.5, 0.5])

        xp = R[0] + np.cos(razim) * np.cos(relev) * self.dist
        yp = R[1] + np.sin(razim) * np.cos(relev) * self.dist
        zp = R[2] + np.sin(relev) * self.dist
        E = np.array((xp, yp, zp))

        self.eye = E
        self.vvec = R - E
        self.vvec = self.vvec / proj3d.mod(self.vvec)

        if abs(relev) > np.pi/2:
            # upside down
            V = np.array((0, 0, -1))
        else:
            V = np.array((0, 0, 1))
        zfront, zback = -self.dist, self.dist

        viewM = proj3d.view_transformation(E, R, V)
        perspM = proj3d.persp_transformation(zfront, zback)
        M0 = np.dot(viewM, np.dot(aspectM, worldM)) ##
        M = np.dot(perspM, M0)
        return M
    return get_proj