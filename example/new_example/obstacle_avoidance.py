#%%  --------------------------------------------------------------------------------------------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from policy_transportation import GaussianProcess


class ObstacleAvoid():
    def __init__(self, center, X_cap):
        super(ObstacleAvoid, self).__init__()
        self.center = center
        self.X_cap = X_cap

        kernel_transport=C(constant_value=10)  * RBF(4*np.ones(2)) + WhiteKernel(0.01)
        self.GP = GaussianProcess(kernel=kernel_transport)
    
    def apply_transportation(self):
        Y_cap = self.GP.predict(self.X_cap, return_std=False)
        X_tilta = self.X_cap + self.alpha(Y_cap - self.X_cap) * (Y_cap - self.X_cap)
        return X_tilta
    
    # @property
    def alpha(self, z):
        dist = np.linalg.norm(self.X_cap - self.center, axis=1)
        return (1/np.exp(dist))[:,np.newaxis]
    
    def fit_GP(self):
        # delta = self.outer_boundary_points - self.inner_boundary_points
        self.GP.fit(self.inner_boundary_points, self.outer_boundary_points)
