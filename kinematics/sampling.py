import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from geomdl import BSpline
from kinematics.configs import KinematicsConfig
from kinematics.utils import make_get_proj

class TrajectorySampler:

    def __init__(self, configs):

        self._curve = BSpline.Curve()
        self._step_size = configs['step_size']
        self._num_samples = configs['num_samples']
        self._vars = configs['vars']
        self._aug_vars = configs['aug_vars']
        self._std_thresh = configs['std_thresh']
        self._poly_degree = configs['poly_degree']

        self._curve.degree = configs['bspline_degree']
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
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
            fig = plt.figure()
            ax = plt.axes(projection ='3d')
            ax.w_xaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
            ax.w_yaxis.set_pane_color((0.7, 0.7, 0.7, 1.0))
            ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))

            ax.plot(x_hist, y_hist, np.zeros(len(x_hist)), '.', color='yellow', markersize=2, lw=0.5, label='history')

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
                ax.plot(x_hist, p(x_hist), np.zeros(len(x_hist)), 'r-', lw=2)
                ax.plot(xs, p(xs), np.zeros(len(xs)),  'r-', lw=2, label='fitted curve')

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
        ctrls = []
        ptss = []
        probs = []
        end_points = []
        for i in range(self._num_samples):
            pts, tr, ctrl = self.sample_cubic_bspline( (x_hist, y_hist), ext_tr, dt )
            probs.append( self.traj_distance(pts, ext_pts) )
            ptss.append(pts)
            trs.append(tr)
            ctrls.append(ctrl)
            end_points.append(pts[-1,:])

        probs = probs / np.sum(probs)
        
        if vis:
            lengths = []
            for i in range(pts.shape[0]):
                min_prob = np.min(probs)
                redness = (probs[i]-min_prob) / np.max(probs-min_prob)
                clr = ((1-redness),0,redness)
                ax.plot(ptss[i][:,0], ptss[i][:,1], np.zeros(ptss[i].shape[0]), color=clr, lw=0.6, alpha=0.3)
                # ax.text(ptss[i][-1,0], ptss[i][-1,1], 0, '{:d}%'.format(int(probs[i]*100)), color=clr, fontsize=7)
                
                lengths.append(np.sum( (ptss[i][1:] - ptss[i][:-1])**2 ))
               
            idx = np.argmax(lengths)
            ax.plot(ptss[idx][:,0], ptss[idx][:,1], np.zeros(ptss[idx].shape[0]), color=(redness,0,(1-redness)), lw=1, alpha=1)
            ax.plot(ctrls[idx][:,0], ctrls[idx][:,1], np.zeros(ctrls[idx].shape[0]), '.', color='black', lw=1)
            ax.plot(ctrls[idx][:,0], ctrls[idx][:,1], np.zeros(ctrls[idx].shape[0]), color='black', lw=1, label='control polygon')
            for l in range(ctrls[idx].shape[0]):
                ax.text(ctrls[idx][l,0]+0.8, ctrls[idx][l,1]+0.8, 0, '$c_{:d}$'.format(l), color='black', fontsize=10)
            
            ax.grid(False)
            ax.legend()
            ax.view_init(elev=45., azim=60)
            ax.get_proj = make_get_proj(ax, 2, 2, 1)

            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
            plt.savefig('sample.pdf', format="pdf")

        return end_points, probs.tolist(), ptss, []

    def sample_cubic_bspline(self, hist, tr_last, dt):
        x_hist = hist[0]
        y_hist = hist[1]

        tr_new = tr_last.copy()
        tr_new[0,:] += np.random.normal(0.0, self._vars[0]*dt, 2)
        tr_new[1,:] += np.random.normal(0.0, self._vars[1]*dt, 2)
        tr_new[2,:] += np.random.normal(0.0, self._vars[2]*dt, 2)

        # idx = int( len(x_hist)/2 )
        idx = np.min((7, len(x_hist)))
        self._curve.ctrlpts = [[x_hist[-idx], y_hist[-idx], 0], [x_hist[-1], y_hist[-1], 0], 
                               [tr_new[0,0], tr_new[0,1], 0], [tr_new[1,0], tr_new[1,1], 0], 
                               [tr_new[2,0], tr_new[2,1], 0]]

        self._curve.knotvector = [0, 0, 0, 0, 0, 1, 1, 1, 1]        
        self._curve_points = self._curve.evalpts

        self._curve_points = np.array( [[p[0],p[1]] for p in self._curve_points] )

        return self._curve_points, tr_new, np.array(self._curve.ctrlpts)

if __name__ == "__main__":
    configs = KinematicsConfig()
    ts = TrajectorySampler(configs.configs)

    x_hist = np.linspace(0, 4, 40)
    x_hist = np.flipud(x_hist)
    x_hist = x_hist.reshape((x_hist.shape[0],1))
    y_hist = -0.4*x_hist**2 + 0.1*x_hist**3 
    y_hist += np.random.rand(x_hist.shape[0],1)*0.25

    ts.sample_gaussian_trajectories( np.concatenate(( x_hist, y_hist, np.zeros(x_hist.shape) ), axis=1), 0.8, vis=True )

