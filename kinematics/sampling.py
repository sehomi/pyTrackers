import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from geomdl import BSpline

class TrajectorySampler:

    def __init__(self):

        self._curve = BSpline.Curve()
        self._step_size = 0.1

        self._curve.degree = 4
        self._curve.delta = self._step_size

    def traj_distance(self, trj1, trj2):
        min_len = np.min([len(trj1), len(trj2)])
        dist = np.sum( np.sqrt( np.sum( (trj2[0:min_len,:] - trj1[0:min_len,:])**2, axis=1) ) )
        return dist

    def sample_gaussian_trajectories(self, poses):
        x_hist = poses[:,0]
        y_hist = poses[:,1]

        fig, ax = plt.subplots()

        ax.plot(x_hist, y_hist, 'y-', lw=2, label='history')

        f = np.polyfit(x_hist, y_hist, 2)
        p = np.poly1d(f)
        p_prime = p.deriv()
        
        s1 = np.array([x_hist[-1] - x_hist[-2], p(x_hist[-1]) - p(x_hist[-2]) ]) 
        s1 = s1/np.linalg.norm(s1)

        s2 = np.array([1, p_prime(x_hist[-1])])
        s2 = s2/np.linalg.norm(s2)

        sgn = np.sign( np.dot(s1, s2) )

        ds = 0
        x = x_hist[-1]
        xs = [x_hist[-1]]
        ext_tr = []
        while ds < 3:
            dx = sgn * np.cos( np.arctan( p_prime(x) ) )*self._step_size
            ds += self._step_size
            x += dx
            xs.append(x)

            if ds >= 1 and len(ext_tr) == 0:
                ext_tr.append([x, p(x)])
            elif ds >= 2 and len(ext_tr) == 1:
                ext_tr.append([x, p(x)])
            elif ds >= 3 and len(ext_tr) == 2:
                ext_tr.append([x, p(x)])
            
        ax.plot(x_hist, p(x_hist), 'r-', lw=2, label='history')
        ax.plot(xs, p(xs), 'r-', lw=2, label='history')

        ext_tr = np.array(ext_tr).astype(np.float64)
        ext_pts = np.concatenate( ( np.array(xs).reshape((len(xs),1)), p(xs).reshape((len(xs),1)) ) , axis=1)

        trs = []
        ptss = []
        probs = []
        for i in range(20):
            pts, tr = self.sample_cubic_bspline( (x_hist, y_hist), ext_tr )
            probs.append( self.traj_distance(pts, ext_pts) )
            ptss.append(pts)
            trs.append(tr)
        probs = probs / np.sum(probs)

        for i in range(pts.shape[0]):
            redness = probs[i] / np.max(probs)
            ax.plot(ptss[i][:,0], ptss[i][:,1], color=(redness,0,(1-redness)), lw=2)

        ax.grid(True)

        plt.savefig('sample.png')

    def sample_cubic_bspline(self, hist, tr_last):
        x_hist = hist[0]
        y_hist = hist[1]

        tr_new = tr_last.copy()
        tr_new[0,:] += np.random.normal(0.0, 1.0, 2)
        tr_new[1,:] += np.random.normal(0.0, 1.5, 2)
        tr_new[2,:] += np.random.normal(0.0, 2.0, 2)

        self._curve.ctrlpts = [[x_hist[-2], y_hist[-2], 0], [x_hist[-1], y_hist[-1], 0], 
                               [tr_new[0,0], tr_new[0,1], 0], [tr_new[1,0], tr_new[1,1], 0], 
                               [tr_new[2,0], tr_new[2,1], 0]]

        self._curve.knotvector = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]        
        self._curve_points = self._curve.evalpts

        self._curve_points = np.array( [[p[0],p[1]] for p in self._curve_points] )

        return self._curve_points, tr_new


ts = TrajectorySampler()

x_hist = np.linspace(0, 4, 40)
x_hist = np.flipud(x_hist)
x_hist = x_hist.reshape((x_hist.shape[0],1))
y_hist = -0.4*x_hist**2 + 0.1*x_hist**3

ts.sample_gaussian_trajectories( np.concatenate(( x_hist, y_hist, np.zeros(x_hist.shape) ), axis=1) )

