from os.path import join
import numpy as np
import pandas as pd
import argparse
import os

from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

class RangeEstimator:

    def __init__(self, img_size, mode='normal'):

        assert mode in ['normal', 'oblique']
        assert len(img_size) >= 2

        self.mode = mode
        model_name = 'model@1535470106'
        model_path = os.path.dirname(__file__) + '/KITTI-distance-estimation/distance-estimator/generated_files'
        data_path = os.path.dirname(__file__) + '/KITTI-distance-estimation/distance-estimator/data'

        self.img_w = img_size[0]
        self.img_h = img_size[1]

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

    def findRange(self, rect, cls='person'):
        assert cls == 'person'

        x1 = np.array([[rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]])
        # x2 = self.scalar1.fit_transform(x1)
        x2 = np.zeros((1,4), dtype=np.float32)
        x2[0,0::2] = x1[0,0::2].astype(np.float32)/self.img_w - 0.5
        x2[0,1::2] = x1[0,1::2].astype(np.float32)/self.img_h - 0.5

        y_pred1 = self.model.predict(x2, verbose = 0)
        y_pred2 = self.scalar2.inverse_transform(y_pred1)
        # print(x1, x2, y_pred1, y_pred2)

        return y_pred2[0][0]
