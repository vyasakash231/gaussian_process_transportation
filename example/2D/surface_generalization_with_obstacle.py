#%%  --------------------------------------------------------------------------------------------------------------------------------------------
import sys
import os
import scipy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import *


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



#%% Load the drawings -------------------------------------------------------------------------------------------------------------------------
source_path = str(pathlib.Path(__file__).parent.absolute())  
data =np.load(source_path+ '/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
X=resample(X, num_points=400)
source_distribution=resample(S, num_points=20)  # (20,2)
target_distribution=resample(S1, num_points=20)  # (20,2)


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

fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
plt.scatter(source_distribution[:,0],source_distribution[:,1], color=[0,1,0])   
plt.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,1]) 
plt.legend(["Demonstration","Surface","New Surface"])



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
plot_vector_field(gp_deltaX1, x1_grid, y1_grid, X1, target_distribution)

"""-------------------  Introduce an Obstacle -------------------------"""
boundary_points = np.array([
    [5, 40],  # bottom-left
    [10, 30],   # bottom-right
    [30, 40],    # top-right
    [25, 50]    # top-left
])

# Generate some points inside the obstacle
num_points = 200
points_inside = sample_in_polygon_convex(boundary_points, num_points)

projected_points = radial_projection(points_inside, boundary_points)

# Plot interior points smaller and more transparent
plt.scatter(points_inside[:, 0], points_inside[:, 1], c='cyan', alpha=0.5, s=20, label='Interior Points')

# Plot boundary points larger and more visible
plt.scatter(projected_points[:, 0], projected_points[:, 1], c='black', label='Boundary Points')

plt.show()



# %%  Transport the dynamical system on the new surface
k_transport = C(constant_value=10)  * RBF(4*np.ones(2)) + WhiteKernel(0.01)

# This is a GP_model_1(x_label = X, y_label = X̂), to map (demo deta = X) to (transported demo deta = X̂)
transport_1=Transport(kernel_transport=k_transport)  
transport_1.source_distribution=source_distribution  # pass source distribution (S)
transport_1.target_distribution=target_distribution  # pass target distribution (τ)
transport_1.training_traj=X  # pass X (demo data subset) into the GP_model_1
transport_1.training_delta=deltaX  # pass Ẋ = ΔX into the GP_model_1

print('Transporting the dynamical system on the new surface')
transport_1.fit_transportation(do_scale=False, do_rotation=True)
transport_1.apply_transportation()
X1=transport_1.training_traj  # we will get X̂ = GP_model_1(X)
deltaX1=transport_1.training_delta # we will get ΔX̂ = GP_model_1(ΔX)

"""---------------------------------- Introduce an Obstacle -------------------------------------"""

# boundary_points = np.array([
#     [5, 40],  # bottom-left
#     [10, 30],   # bottom-right
#     [30, 40],    # top-right
#     [25, 50]    # top-left
# ])

boundary_points = np.array([
    [5, 40],  # bottom-left
    [5, 30],   # bottom-right
    [30, 30],    # top-right
    [30, 40]    # top-left
])

# Generate some points inside the obstacle
num_points = 50
num_contours = 3

# points_inside = sample_in_polygon(boundary_points, num_points)

# projected_points = radial_projection(points_inside, boundary_points)
# projected_points = sdf_projection(points_inside, boundary_points)

inter_boundary_points = generate_inner_contours(boundary_points, num_points, num_contours)   # (num_points x (num_contours+1), 2)

# target_distribution = np.vstack((inter_boundary_points[:50], inter_boundary_points[:50], inter_boundary_points[:50]))
source_distribution = inter_boundary_points[50:]
projected_points = radial_projection(source_distribution, boundary_points)

# This is a GP_model_1(x_label = X, y_label = X̂), to map (demo deta = X) to (transported demo deta = X̂)
transport_2=Transport(kernel_transport=k_transport)  
transport_2.source_distribution=source_distribution  # pass source distribution (S)
transport_2.target_distribution=projected_points  # pass target distribution (τ)
transport_2.training_traj=X1  # pass X (demo data subset) into the GP_model_1
transport_2.training_delta=deltaX1  # pass Ẋ = ΔX into the GP_model_1

print('Transporting the dynamical system with obstacle on the new surface')
transport_2.fit_transportation(do_scale=False, do_rotation=True)  # do_scale will scale the entire state-space and trajectory
transport_2.apply_transportation()
X2=transport_2.training_traj  # we will get X̂ = GP_model_1(X)
deltaX2=transport_2.training_delta # we will get ΔX̂ = GP_model_1(ΔX)

# Fit the Gaussian Process dynamical system   
print('Fitting the GP dynamical system on the transported trajectory')
k_deltaX2 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01)    
gp_deltaX2=GPR(kernel=k_deltaX2)
gp_deltaX2.fit(X2, deltaX2)  # fit a new GP_model_2(x_label = ΔX, y_label = ΔX̂)
x2_grid=np.linspace(np.min(X2[:,0]-10), np.max(X2[:,0]+10), 60)
y2_grid=np.linspace(np.min(X2[:,1]-10), np.max(X2[:,1]+10), 60)
dataXX, dataYY = np.meshgrid(x2_grid, y2_grid)
pos_array = np.column_stack((dataXX.ravel(), dataYY.ravel()))

vel = gp_deltaX2.predict(pos_array)
u = vel[:, 0].reshape(dataXX.shape)
v = vel[:, 1].reshape(dataXX.shape)
"""----------------------------- Plot ---------------------------------"""
fig, ax = plt.figure(figsize=(12, 12)), plt.gca()
ax.set_aspect(1)
ax.quiver(dataXX, dataYY, u, v, color='blue', alpha=0.6)
ax.scatter(X2[:, 0], X2[:, 1], color=[1, 0, 0])
ax.scatter(projected_points[:, 0], projected_points[:, 1], color=[0, 0, 0])

# Plot interior points smaller and more transparent
# plt.scatter(points_inside[:, 0], points_inside[:, 1], c='cyan', alpha=0.5, s=20, label='Interior Points')

# Plot boundary points larger and more visible
# plt.scatter(projected_points[:, 0], projected_points[:, 1], c='Magenta', label='Boundary Points')
plt.show()





# %%  Transport the dynamical system on the new surface
k_transport = C(constant_value=10)  * RBF(4*np.ones(2)) + WhiteKernel(0.01)

# This is a GP_model_1(x_label = X, y_label = X̂), to map (demo deta = X) to (transported demo deta = X̂)
transport_1=Transport(kernel_transport=k_transport)  
transport_1.source_distribution=source_distribution  # pass source distribution (S)
transport_1.target_distribution=target_distribution  # pass target distribution (τ)
transport_1.training_traj=X  # pass X (demo data subset) into the GP_model_1
transport_1.training_delta=deltaX  # pass Ẋ = ΔX into the GP_model_1

print('Transporting the dynamical system on the new surface')
transport_1.fit_transportation(do_scale=False, do_rotation=True)
transport_1.apply_transportation()
X1=transport_1.training_traj  # we will get X̂ = GP_model_1(X)
deltaX1=transport_1.training_delta # we will get ΔX̂ = GP_model_1(ΔX)

"""---------------------------------- Introduce an Obstacle -------------------------------------"""

# boundary_points = np.array([
#     [5, 40],  # bottom-left
#     [10, 30],   # bottom-right
#     [30, 40],    # top-right
#     [25, 50]    # top-left
# ])

boundary_points = np.array([
    [5, 40],  # bottom-left
    [5, 30],   # bottom-right
    [30, 30],    # top-right
    [30, 40]    # top-left
])

# Generate some points inside the obstacle
num_points = 200
# points_inside = sample_in_polygon_convex(boundary_points, num_points)
points_inside = sample_in_polygon(boundary_points, num_points)
projected_points = radial_projection(points_inside, boundary_points)

# This is a GP_model_1(x_label = X, y_label = X̂), to map (demo deta = X) to (transported demo deta = X̂)
transport_2=Transport(kernel_transport=k_transport)  
transport_2.source_distribution=points_inside  # pass source distribution (S)
transport_2.target_distribution=projected_points  # pass target distribution (τ)
transport_2.training_traj=X1  # pass X (demo data subset) into the GP_model_1
transport_2.training_delta=deltaX1  # pass Ẋ = ΔX into the GP_model_1

print('Transporting the dynamical system with obstacle on the new surface')
transport_2.fit_transportation(do_scale=False, do_rotation=True)  # do_scale will scale the entire state-space and trajectory
transport_2.apply_transportation()
X2=transport_2.training_traj  # we will get X̂ = GP_model_1(X)
deltaX2=transport_2.training_delta # we will get ΔX̂ = GP_model_1(ΔX)

# Fit the Gaussian Process dynamical system   
print('Fitting the GP dynamical system on the transported trajectory')
k_deltaX2 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01)    
gp_deltaX2=GPR(kernel=k_deltaX2)
gp_deltaX2.fit(X2, deltaX2)  # fit a new GP_model_2(x_label = ΔX, y_label = ΔX̂)

# add extra points for GP model training
# velocity = generate_divergent_rotational_flow(boundary_points, points_inside)
velocity = generate_shaped_divergent_flow(boundary_points, points_inside)

gp_deltaX2.fit(np.vstack((X2, points_inside)), np.vstack((deltaX2, velocity)))  # fit a new GP_model_2(x_label = ΔX, y_label = ΔX̂)
x2_grid=np.linspace(np.min(X2[:,0]-10), np.max(X2[:,0]+10), 60)
y2_grid=np.linspace(np.min(X2[:,1]-10), np.max(X2[:,1]+10), 60)
dataXX, dataYY = np.meshgrid(x2_grid, y2_grid)
pos_array = np.column_stack((dataXX.ravel(), dataYY.ravel()))

vel = gp_deltaX2.predict(pos_array)
u = vel[:, 0].reshape(dataXX.shape)
v = vel[:, 1].reshape(dataXX.shape)
"""----------------------------- Plot ---------------------------------"""
fig, ax = plt.figure(figsize=(12, 12)), plt.gca()
ax.set_aspect(1)
ax.quiver(dataXX, dataYY, u, v, color='blue', alpha=0.6)
ax.scatter(X2[:, 0], X2[:, 1], color=[1, 0, 0])
ax.scatter(projected_points[:, 0], projected_points[:, 1], color=[0, 0, 0])

# Plot interior points smaller and more transparent
plt.scatter(points_inside[:, 0], points_inside[:, 1], c='cyan', alpha=0.5, s=20, label='Interior Points')

# Plot boundary points larger and more visible
# plt.scatter(projected_points[:, 0], projected_points[:, 1], c='Magenta', label='Boundary Points')
plt.show()





# %%  Transport the dynamical system on the new surface
k_transport = C(constant_value=10)  * RBF(4*np.ones(2)) + WhiteKernel(0.01)

# This is a GP_model_1(x_label = X, y_label = X̂), to map (demo deta = X) to (transported demo deta = X̂)
transport_1=Transport(kernel_transport=k_transport)  
transport_1.source_distribution=source_distribution  # pass source distribution (S)
transport_1.target_distribution=target_distribution  # pass target distribution (τ)
transport_1.training_traj=X  # pass X (demo data subset) into the GP_model_1
transport_1.training_delta=deltaX  # pass Ẋ = ΔX into the GP_model_1

print('Transporting the dynamical system on the new surface')
transport_1.fit_transportation(do_scale=False, do_rotation=True)
transport_1.apply_transportation()
X1=transport_1.training_traj  # we will get X̂ = GP_model_1(X)
deltaX1=transport_1.training_delta # we will get ΔX̂ = GP_model_1(ΔX)

# Fit the Gaussian Process dynamical system   
print('Fitting the GP dynamical system on the transported trajectory')
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01)    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)  # fit a new GP_model_2(x_label = ΔX, y_label = ΔX̂)
x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 60)
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 60)
dataXX, dataYY = np.meshgrid(x1_grid, y1_grid)
pos_array = np.column_stack((dataXX.ravel(), dataYY.ravel()))
vel = gp_deltaX1.predict(pos_array)

"""---------------------------------- Introduce an Obstacle -------------------------------------"""

boundary_points = np.array([
    [5, 40],  # bottom-left
    [10, 30],   # bottom-right
    [30, 40],    # top-right
    [25, 50]    # top-left
])

# Generate some points inside the obstacle
num_points = 100
# points_inside = sample_in_polygon_convex(boundary_points, num_points)
points_inside = sample_in_polygon(boundary_points, num_points)

# Create flow field
flow_field = ObstacleFlowField(boundary_points)

# Learn and visualize flow field
flow_field.learn_flow_field(points_inside)

# Transform the flow field
X2, _ = flow_field.transform_space(X1)
transformed_grid, _ = flow_field.transform_space(pos_array)
transformed_grid_vel = flow_field.transform_velocity(pos_array, vel)

"""----------------------------- Plot ---------------------------------"""
XX = transformed_grid[:, 0]
YY = transformed_grid[:, 1]

u = transformed_grid_vel[:, 0]
v = transformed_grid_vel[:, 1]

# Plot
fig, ax = plt.figure(figsize=(12, 12)), plt.gca()
ax.set_aspect(1)
ax.quiver(XX, YY, u, v, color='blue', alpha=0.6)

ax.scatter(X2[:, 0], X2[:, 1], color=[1, 0, 0])

# Plot interior points smaller and more transparent
plt.scatter(points_inside[:, 0], points_inside[:, 1], c='cyan', alpha=0.5, s=20, label='Interior Points')

# Plot boundary points larger and more visible
ax.scatter(flow_field.projected_boundary_points[:, 0], flow_field.projected_boundary_points[:, 1], c='black', label='Boundary Points')
plt.show()



# %%
