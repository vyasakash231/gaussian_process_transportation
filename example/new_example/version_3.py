#%%  --------------------------------------------------------------------------------------------------------------------------------------------
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
from policy_transportation.plot_utils import *
from policy_transportation.utils import resample
import warnings
warnings.filterwarnings("ignore")

from utils import *
from obstacle_avoidance import ObstacleAvoid

#%% Load the drawings -------------------------------------------------------------------------------------------------------------------------
source_path = str(pathlib.Path(__file__).parent.absolute())  
data =np.load(source_path+ '/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
X=resample(X, num_points=400)
source_distribution=resample(S, num_points=20)  # (20,2)
target_distribution=resample(S1, num_points=20)  # (20,2)

# theta = np.linspace(np.pi/4, 5*np.pi/4, 20)
# # theta = np.linspace(0, 3*np.pi/2, 20)
# x = 20 + 20*np.cos(theta)
# y = 0 + 10*np.sin(theta) + np.random.normal(0, 0.25, 20)
# target_distribution = np.vstack((x, y)).T

# x = np.linspace(15, 30, 20)
# y = np.linspace(-10, 20, 20) #+ np.random.normal(0, 0.5, 20)
# target_distribution = np.vstack((x, y)).T


#%% Calculate deltaX --------------------------------------------------------------------------------------------------------------------------
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])



#%% Fit a dynamical system (X_dot = f(X)) to the demo and plot it ----------------------------------------------------------------------------
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01) 
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)
x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
plot_vector_field(gp_deltaX, x_grid, y_grid, X, target_distribution)

# fig = plt.figure(figsize = (12, 7)), plt.gca()
# plt.xlim([-50, 50-1])
# plt.ylim([-50, 50-1])
plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
plt.scatter(source_distribution[:,0],source_distribution[:,1], color=[0,1,0])   
plt.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,1]) 
# plt.legend(["Demonstration","Surface","New Surface"])



#%% Transport the dynamical system on the new surface -----------------------------------------------------------------------------------------
k_transport = C(constant_value=10)  * RBF(4*np.ones(2)) + WhiteKernel(0.01)

# This is a GP_model_1(x_label = X, y_label = X̂), to map (demo deta = X) to (transported demo deta = X̂)
transport=Transport(kernel_transport=k_transport)  
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


# %%  Transport the dynamical system on the new surface
k_transport = C(constant_value=10)  * RBF(4*np.ones(2)) + WhiteKernel(0.01)

# This is a GP_model_1(x_label = X, y_label = X̂), to map (demo deta = X) to (transported demo deta = X̂)
transport=Transport(kernel_transport=k_transport)  
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

dataXX, dataYY = np.meshgrid(x1_grid, y1_grid)
pos_array = np.column_stack((dataXX.ravel(), dataYY.ravel()))

vel = gp_deltaX1.predict(pos_array)

"""----------------------------- Plot ---------------------------------"""
u = vel[:, 0].reshape(dataXX.shape)
v = vel[:, 1].reshape(dataXX.shape)

fig, ax = plt.figure(figsize=(12, 7)), plt.gca()
ax.set_aspect(1)
ax.streamplot(dataXX, dataYY, u, v, density=2)
ax.scatter(X1[:, 0], X1[:, 1], color=[1, 0, 0])

"""-------------------  Introduce an Obstacle -------------------------"""
theta = np.linspace(0, 2*np.pi, 20)
# theta = np.linspace(0, 3*np.pi/2, 20)
x = 25 + 10*np.cos(theta)
y = 30 + 5*np.sin(theta) + np.random.normal(0, 0.25, 20)
obstacle_boundary_points = np.vstack((x, y)).T

# Plot boundary points larger and more visible
plt.scatter(x, y, c='black', label='obstacle Boundary Points')
plt.show()



# %%  Transport the dynamical system on the new surface
k_transport = C(constant_value=10)  * RBF(4*np.ones(2)) + WhiteKernel(0.01)

# This is a GP_model_1(x_label = X, y_label = X̂), to map (demo deta = X) to (transported demo deta = X̂)
transport=Transport(kernel_transport=k_transport)  
transport.source_distribution=source_distribution  # pass source distribution (S)
transport.target_distribution=target_distribution  # pass target distribution (τ)
transport.training_traj=X  # pass X (demo data subset) into the GP_model_1
transport.training_delta=deltaX  # pass Ẋ = ΔX into the GP_model_1

print('Transporting the dynamical system on the new surface')
transport.fit_transportation(do_scale=False, do_rotation=True)
transport.apply_transportation()
X1=transport.training_traj  # we will get X̂ = GP_model_1(X)
deltaX1=transport.training_delta # we will get ΔX̂ = GP_model_1(ΔX)

"""-------------------  Introduce an Obstacle -------------------------"""
theta = np.linspace(0, 2*np.pi, 20)
x_inner = 25 + 10*np.cos(theta)
y_inner = 30 + 5*np.sin(theta)
inner_boundary_points = np.vstack((x_inner, y_inner)).T

theta = np.linspace(np.pi/2, 3*np.pi/2, 20)
x_outer = 25 + 12*np.cos(theta)
y_outer = 30 + 6*np.sin(theta) + np.random.normal(0, 0.25, 20)
outer_boundary_points = np.vstack((x_outer, y_outer)).T

obstacle_def = ObstacleCenterEstimator(inner_boundary_points)
obstacle_center, _ = obstacle_def.estimate_with_pca()
first, second = obstacle_def.estimate_dimensions()
# n_vec = outer_boundary_points - obstacle_center
# unit_vec = n_vec / np.linalg.norm(n_vec, axis=1)[:, np.newaxis]


# This is a GP_model_1(x_label = X, y_label = X̂), to map (demo deta = X) to (transported demo deta = X̂)
transport_2=ObstacleAvoid(obstacle_center, X1)  
transport_2.inner_boundary_points=inner_boundary_points  # pass source distribution (S)
transport_2.outer_boundary_points=outer_boundary_points  # pass target distribution (τ)
transport_2.fit_GP()

X2=transport_2.apply_transportation()
# deltaX2=transport_2.training_delta # we will get ΔX̂ = GP_model_1(ΔX)

# print('Fitting the GP dynamical system on the transported trajectory')
# k_deltaX2 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01)    
# gp_deltaX2=GPR(kernel=k_deltaX2)
# gp_deltaX2.fit(X2, deltaX2)  # fit a new GP_model_2(x_label = ΔX, y_label = ΔX̂)

x1_grid=np.linspace(-70, 60, 100)
y1_grid=np.linspace(-70, 60, 100)

dataXX, dataYY = np.meshgrid(x1_grid, y1_grid)
pos_array = np.column_stack((dataXX.ravel(), dataYY.ravel()))

# vel = gp_deltaX2.predict(pos_array)


# """----------------------------- Plot ---------------------------------"""
# u = vel[:, 0].reshape(dataXX.shape)
# v = vel[:, 1].reshape(dataXX.shape)

fig, ax = plt.figure(figsize=(12, 7)), plt.gca()
ax.set_aspect(1)
# ax.streamplot(dataXX, dataYY, u, v, density=2)
ax.scatter(X2[:, 0], X2[:, 1], color=[1, 0, 0])
# # ax.quiver(np.tile(obstacle_center[0],20), np.tile(obstacle_center[1],20), unit_vec[:,0], unit_vec[:,1], color='green')

# # Plot boundary points larger and more visible
plt.scatter(x_inner, y_inner, c='black', label='obstacle Boundary Points')
# # plt.scatter(x_outer, y_outer, c='black', label='obstacle Boundary Points')

plt.show()

# %%
