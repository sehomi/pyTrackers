import numpy as np
import matplotlib.pyplot as plt
from geomdl import BSpline

class TrajectorySampler:

    def __init__(self):

        self._curve = BSpline.Curve()
        self._curve.degree = 4
        self._curve.delta = 0.05

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


fig, ax = plt.subplots()

ts = TrajectorySampler()

x_hist = np.linspace(0, 4, 40)
x_hist = np.flipud(x_hist)
y_hist = -0.4*x_hist**2 + 0.1*x_hist**3

ax.plot(x_hist, y_hist, 'y-', lw=2, label='history')

ext_tr = np.array([[-2,-1],[-3,-1], [-4,0]]).astype(np.float64)

for i in range(20):
    if i==0:
        pts, tr = ts.sample_cubic_bspline( (x_hist, y_hist), ext_tr )
    else:
        pts, tr = ts.sample_cubic_bspline( (x_hist, y_hist), tr )
        
    ax.plot(pts[:,0], pts[:,1], lw=2)

ax.grid(True)

plt.savefig('sample.png')