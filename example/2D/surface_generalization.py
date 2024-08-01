"""
Authors:  Giovanni Franzese and Ravi Prakash, Dec 2022
Email: g.franzese@tudelft.nl, r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""


#%%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from policy_transportation import GaussianProcess as GPR
from policy_transportation import GaussianProcessTransportation as Transport
# from policy_transportation.transportation.gaussian_process_transportation_diffeomorphic import GaussianProcessTransportationDiffeo as Transport
import pathlib
from policy_transportation.plot_utils import plot_vector_field, plot_modified_vector_field_1, plot_modified_vector_field_2, plot_vector_field_minvar
from policy_transportation.utils import resample
import warnings
warnings.filterwarnings("ignore")

#%% Load the drawings
source_path = str(pathlib.Path(__file__).parent.absolute())  
data =np.load(source_path+ '/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
X=resample(X, num_points=100)
source_distribution=resample(S, num_points=20)
target_distribution=resample(S1, num_points=20)

#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])

# #%% Fit a dynamical system (X_dot = f(X)) to the demo and plot it
# k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01) 
# gp_deltaX=GPR(kernel=k_deltaX)
# gp_deltaX.fit(X, deltaX)
# x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
# y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
# plot_vector_field(gp_deltaX, x_grid, y_grid, X, target_distribution)

# fig = plt.figure(figsize = (12, 7))
# plt.xlim([-50, 50-1])
# plt.ylim([-50, 50-1])
# plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
# plt.scatter(source_distribution[:,0],source_distribution[:,1], color=[0,1,0])   
# plt.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,1]) 
# plt.legend(["Demonstration","Surface","New Surface"])

#%% Transport the dynamical system on the new surface
k_transport = C(constant_value=10)  * RBF(4*np.ones(2)) + WhiteKernel(0.01 )

# This is a GP_model_1(x_label = X, y_label = X̂), to map (demo deta = X) to (transported demo deta = X̂)
transport=Transport(kernel_transport= k_transport)  
transport.source_distribution=source_distribution  # pass source distribution (S)
transport.target_distribution=target_distribution  # pass target distribution (τ)
transport.training_traj=X  # pass X (demo data subset) into the GP_model_1
transport.training_delta=deltaX  # pass Ẋ = ΔX into the GP_model_1

print('Transporting the dynamical system on the new surface')
transport.fit_transportation(do_scale=False, do_rotation=True)
transport.apply_transportation()
X1=transport.training_traj  # we will get X̂ = GP_model_1(X)
deltaX1=transport.training_delta # we will get ΔX̂ = GP_model_1(ΔX)

# Fit the Gaussian Process dynamical system   
print('Fitting the GP dynamical system on the transported trajectory')
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01)    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)  # fit a new GP_model_2(x_label = ΔX, y_label = ΔX̂)
x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 100)
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 100)
plot_vector_field(gp_deltaX1, x1_grid, y1_grid, X1, target_distribution)
plt.show()

#%% Transport the dynamical system on the new surface
k_transport = C(constant_value=10)  * RBF(4*np.ones(2)) + WhiteKernel(0.01 )

# This is a GP_model_1(x_label = X, y_label = X̂), to map (demo deta = X) to (transported demo deta = X̂)
transport=Transport(kernel_transport= k_transport)  
transport.source_distribution=source_distribution  # pass source distribution (S)
transport.target_distribution=target_distribution  # pass target distribution (τ)
transport.training_traj=X  # pass X (demo data subset) into the GP_model_1
transport.training_delta=deltaX  # pass Ẋ = ΔX into the GP_model_1

print('Transporting the dynamical system on the new surface')
transport.fit_transportation(do_scale=False, do_rotation=True)
transport.apply_transportation()
X1=transport.training_traj  # we will get X̂ = GP_model_1(X)
deltaX1=transport.training_delta # we will get ΔX̂ = GP_model_1(ΔX)

# Spherical Obstacle 
obstacle_center = np.array([[-20], [30]])
radius = 5

# Fit the Gaussian Process dynamical system   
print('Fitting the GP dynamical system on the transported trajectory')
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01)    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)  # fit a new GP_model_2(x_label = ΔX, y_label = ΔX̂)
x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 100)  # 100 x-samples
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 100)  # 100 y-samples
plot_modified_vector_field_1(gp_deltaX1, x1_grid, y1_grid, X1, target_distribution,
                           obstacle_center, radius)
plt.show()

#%% Transport the dynamical system on the new surface
k_transport = C(constant_value=10)  * RBF(4*np.ones(2)) + WhiteKernel(0.01 )

# This is a GP_model_1(x_label = X, y_label = X̂), to map (demo deta = X) to (transported demo deta = X̂)
transport=Transport(kernel_transport= k_transport)  
transport.source_distribution=source_distribution  # pass source distribution (S)
transport.target_distribution=target_distribution  # pass target distribution (τ)
transport.training_traj=X  # pass X (demo data subset) into the GP_model_1
transport.training_delta=deltaX  # pass Ẋ = ΔX into the GP_model_1

print('Transporting the dynamical system on the new surface')
transport.fit_transportation(do_scale=False, do_rotation=True)
transport.apply_transportation()
X1=transport.training_traj  # we will get X̂ = GP_model_1(X)
deltaX1=transport.training_delta # we will get ΔX̂ = GP_model_1(ΔX)

# Eliptic Obstacle
obstacle_center = np.array([[-20], [30]])
r1 = 4
r2 = 6
m = 4

# Fit the Gaussian Process dynamical system   
print('Fitting the GP dynamical system on the transported trajectory')
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01)    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)  # fit a new GP_model_2(x_label = ΔX, y_label = ΔX̂)
x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 100)  # 100 x-samples
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 100)  # 100 y-samples
plot_modified_vector_field_2(gp_deltaX1, x1_grid, y1_grid, X1, target_distribution,
                           obstacle_center, r1, r2, m)
plt.show()
# %%
