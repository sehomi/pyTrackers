import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from geomdl import BSpline

class TrajectorySampler:

    def __init__(self):

        self._curve = BSpline.Curve()
        self._step_size = 0.1
        self._num_samples = 10
        self._vars = [0.5, 1.5, 3]
        self._aug_vars = 0.8
        self._std_thresh = 0.5
        self._poly_degree = 1

        self._curve.degree = 4
        self._curve.delta = self._step_size

    def traj_distance(self, trj1, trj2):
        min_len = np.min([len(trj1), len(trj2)])
        dist = np.sum( np.sqrt( np.sum( (trj2[0:min_len,:] - trj1[0:min_len,:])**2, axis=1) ) )
        return dist

    def create_augmented_poses(self, poses):
        aug_data = np.random.normal(0.0, self._aug_vars, poses.shape)
        aug_data[:,2] = 0

        return poses+aug_data

    def sample_gaussian_trajectories(self, poses, dt, vis=False):
        if poses.shape[0] < 2:
            res = [np.array([poses[0,0], poses[0,1]])]
            return res, [1], [np.array(res)], []

        x_hist = poses[:,0]
        y_hist = poses[:,1]

        std = np.linalg.norm( [np.std( x_hist ), np.std( y_hist )] )
        mean = [np.mean( x_hist ), np.mean( y_hist )]

        if vis:
            fig, ax = plt.subplots()
            ax.plot(x_hist, y_hist, 'y-', lw=2, label='history')

        aug_poses = []
        if std > self._std_thresh:
            aug_poses = self.create_augmented_poses(poses)
            # x_hist_aug = np.concatenate( (x_hist, aug_poses[:,0] ), axis=0 )
            # y_hist_aug = np.concatenate( (y_hist, aug_poses[:,1] ), axis=0 )
            f = np.polyfit(x_hist, y_hist, self._poly_degree)
            p = np.poly1d(f)
            p_prime = p.deriv()
            
            idx = int( len(x_hist)/2 ) + 1
            s1 = np.array([x_hist[-1] - x_hist[-idx], p(x_hist[-1]) - p(x_hist[-idx]) ]) 
            s1 = s1/np.linalg.norm(s1)

            s2 = np.array([1, p_prime(x_hist[-1])])
            s2 = s2/np.linalg.norm(s2)

            sgn = np.sign( np.dot(s1, s2) )
            if np.isnan(sgn): sgn = 1

            ds = 0
            x = x_hist[-1].copy()
            xs = [x_hist[-1]]
            ext_tr = []
            while ds < self._vars[2]*dt:
                dx = sgn * np.cos( np.arctan( p_prime(x) ) )*self._step_size
                ds += self._step_size
                x += dx
                xs.append(x)

                if ds >= self._vars[0]*dt and len(ext_tr) == 0:
                    ext_tr.append([x, p(x)])
                elif ds >= self._vars[1]*dt and len(ext_tr) == 1:
                    ext_tr.append([x, p(x)])
                elif ds >= self._vars[2]*dt and len(ext_tr) == 2:
                    ext_tr.append([x, p(x)])
                
            if vis:
                ax.plot(x_hist, p(x_hist), 'r-', lw=2, label='history')
                ax.plot(xs, p(xs), 'r-', lw=2, label='history')

            ext_tr = np.array(ext_tr).astype(np.float64)
            ext_pts = np.concatenate( ( np.array(xs).reshape((len(xs),1)), p(xs).reshape((len(xs),1)) ) , axis=1)
            
        else:
            ext_tr = np.array([mean,mean,mean])
            ext_pts = ext_tr.copy()

        # trs = []
        # ptss = [np.array(ext_pts)]
        # probs = [1]
        # end_points = [ptss[0][-1]]

        trs = []
        ptss = []
        probs = []
        end_points = []
        for i in range(self._num_samples):
            pts, tr = self.sample_cubic_bspline( (x_hist, y_hist), ext_tr, dt )
            probs.append( self.traj_distance(pts, ext_pts) )
            ptss.append(pts)
            trs.append(tr)
            end_points.append(pts[-1,:])

        probs = probs / np.sum(probs)
        
        if vis:
            for i in range(pts.shape[0]):
                min_prob = np.min(probs)
                redness = (probs[i]-min_prob) / np.max(probs-min_prob)
                ax.plot(ptss[i][:,0], ptss[i][:,1], color=(redness,0,(1-redness)), lw=2)
            ax.grid(True)

        return end_points, probs.tolist(), ptss, []

    def sample_cubic_bspline(self, hist, tr_last, dt):
        x_hist = hist[0]
        y_hist = hist[1]

        tr_new = tr_last.copy()
        tr_new[0,:] += np.random.normal(0.0, self._vars[0]*dt, 2)
        tr_new[1,:] += np.random.normal(0.0, self._vars[1]*dt, 2)
        tr_new[2,:] += np.random.normal(0.0, self._vars[2]*dt, 2)

        # idx = int( len(x_hist)/2 )
        idx = np.min((3, len(x_hist)))
        self._curve.ctrlpts = [[x_hist[-idx], y_hist[-idx], 0], [x_hist[-1], y_hist[-1], 0], 
                               [tr_new[0,0], tr_new[0,1], 0], [tr_new[1,0], tr_new[1,1], 0], 
                               [tr_new[2,0], tr_new[2,1], 0]]

        self._curve.knotvector = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]        
        self._curve_points = self._curve.evalpts

        self._curve_points = np.array( [[p[0],p[1]] for p in self._curve_points] )

        return self._curve_points, tr_new

if __name__ == "__main__":
    ts = TrajectorySampler()

    x_hist = np.linspace(0, 4, 40)
    x_hist = np.flipud(x_hist)
    x_hist = x_hist.reshape((x_hist.shape[0],1))
    y_hist = -0.4*x_hist**2 + 0.1*x_hist**3

    ts.sample_gaussian_trajectories( np.concatenate(( x_hist, y_hist, np.zeros(x_hist.shape) ), axis=1), vis=True )

