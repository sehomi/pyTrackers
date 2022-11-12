from os.path import join
import numpy as np
import pandas as pd
import argparse
import os

from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

class RangeEstimator:

    def __init__(self, img_size, method='direct', direct_mode='normal', margin=20):

        assert method in ['proportionality', 'direct']
        assert direct_mode in ['normal', 'oblique']
        assert len(img_size) >= 2

        self.method = method
        self.direct_mode = direct_mode
        model_name = 'model@1535470106'
        model_path = os.path.dirname(__file__) + '/KITTI-distance-estimation/distance-estimator/generated_files'
        data_path = os.path.dirname(__file__) + '/KITTI-distance-estimation/distance-estimator/data'

        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.margin = margin
        self.last_z = None

        # load json and create model
        json_file = open('{}/{}.json'.format(model_path, model_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json( loaded_model_json )

        # load weights into new model
        loaded_model.load_weights('{}/{}.h5'.format(model_path, model_name))

        # evaluate loaded model on test data
        loaded_model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = loaded_model

        # get data
        df_test = pd.read_csv('{}/test.csv'.format(data_path))
        X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
        y_test = df_test[['zloc']].values

        # standardized data
        self.scalar1 = StandardScaler()
        X_test = self.scalar1.fit_transform(X_test)
        self.scalar2 = StandardScaler()
        y_test = self.scalar2.fit_transform(y_test)

    def isDistantFromBoundary(self, rect):
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[0] + rect[2]
        y2 = rect[1] + rect[3]

        return x1>self.margin and y1>self.margin and x2<self.img_w-self.margin and y2<self.img_h-self.margin

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

    def findPos(self, rect, direction, z, cls='person'):
        assert cls == 'person'

        pos = None
        if self.method == 'direct':
            if self.isDistantFromBoundary(rect):
                x1 = np.array([[rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]])
                # x2 = self.scalar1.fit_transform(x1)
                x2 = np.zeros((1,4), dtype=np.float32)
                x2[0,0::2] = x1[0,0::2].astype(np.float32)/self.img_w - 0.5
                x2[0,1::2] = x1[0,1::2].astype(np.float32)/self.img_h - 0.5

                y_pred1 = self.model.predict(x2, verbose = 0)
                y_pred2 = self.scalar2.inverse_transform(y_pred1)
                # print(x1, x2, y_pred1, y_pred2)

                rng = y_pred2[0][0]
                pos = rng*direction
            else:
                if self.last_z is not None:
                    pos = self.scale_vector(direction, self.last_z)
                else:
                    pos = self.scale_vector(direction, self.z)

        elif self.method == 'proportionality':
            pos = self.scale_vector(direction, z)

        self.last_z = np.abs(pos[2])

        return pos
