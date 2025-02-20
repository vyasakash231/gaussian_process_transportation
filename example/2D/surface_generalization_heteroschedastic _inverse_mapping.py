"""
Authors:  Giovanni Franzese, Dec 2022
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
import sklearn
if sklearn.__version__<='1.3.0':
    print('scikit-learn version: ', sklearn.__version__)
    print('Please install scikit-learn newer than 1.3.0')
    exit()
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

import matplotlib.pyplot as plt
from policy_transportation import GaussianProcess as GPR
from policy_transportation.transportation.gaussian_process_transportation import GaussianProcessTransportation as Transport
import pathlib
from policy_transportation.plot_utils import plot_vector_field_minvar, plot_vector_field 
import warnings
warnings.filterwarnings("ignore")
from policy_transportation.utils import resample  
#%% Load the drawings
source_path = str(pathlib.Path(__file__).parent.absolute())  
data =np.load(source_path+ '/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
plt.scatter(S[:,0],S[:,1], color=[0,1,0])   
plt.scatter(S1[:,0],S1[:,1], color=[0,0,1]) 
plt.legend(["Demonstration","Surface","New Surface"])

X=resample(X, num_points=200)
source_distribution=resample(S,num_points=100)
target_distribution=resample(S1, num_points=100)
#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])

#%% Fit a dynamical system to the demo and plot it
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01) 
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)
x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
plot_vector_field(gp_deltaX, x_grid,y_grid,X,S)

x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)

X_, Y_ = np.meshgrid(x_grid, y_grid)
input=np.hstack((X_.reshape(-1,1),Y_.reshape(-1,1)))

sigma_noise_prediction=[]
std_gp=gp_deltaX.predict(input)[1][:,0]

std_gp=std_gp.reshape(X_.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_, Y_, std_gp, linewidth=0, antialiased=True, cmap=plt.cm.inferno)
plt.axis('off')

# Create a source and a target distribution with the same number of nodes
# source_distribution=resample(S)
# target_distribution=resample(S1)
#%% Transport the dynamical system on the new surface
transport=Transport()
transport.source_distribution=source_distribution 
transport.target_distribution=target_distribution
transport.training_traj=X
transport.training_delta=deltaX
print('Transporting the dynamical system on the new surface')
k_transport = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(2)) + WhiteKernel(0.01 )
transport.kernel_transport=k_transport
transport.fit_transportation()
transport.apply_transportation()
X1=transport.training_traj
deltaX1=transport.training_delta 


transport_inverse=Transport()
transport_inverse.source_distribution=target_distribution 
transport_inverse.target_distribution=source_distribution
k_transport = C(constant_value=np.sqrt(0.1))  * RBF(1) + WhiteKernel(0.01 )
transport_inverse.kernel_transport=k_transport
print("Calculate Inverse Mapping")
transport_inverse.fit_transportation()
#%% Plot the transported dynamical system
#
x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 100)
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 100)
sigma_noise=[]
# Calculate the noise on the transported trajectory using the epistemic uncertainty of the GP
for i in range(len(X)):
    [Jacobian,_]=transport.gp_delta_map.derivative(X[i].reshape(1,-1))
    _, std= transport.gp_delta_map.predict(X[i].reshape(1,-1))
    var_derivative=std**2/(np.linalg.norm(transport.gp_delta_map.kernel_params_[0]))**2 #this is an approximation of the variance of the derivative
    sigma_noise.append(var_derivative)
sigma_noise=np.array(sigma_noise)
# print(sigma_noise)
# Fit the Gaussian Process dynamical system   
print('Fitting the GP dynamical system on the transported trajectory')
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX1=GPR(kernel=k_deltaX1)
# print(sigma_noise)
gp_deltaX1.fit(X1, deltaX1)
plot_vector_field(gp_deltaX1, x1_grid,y1_grid,X1,S1)
# plt.show()


X, Y = np.meshgrid(x1_grid, y1_grid)
input=np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))

sigma_noise_prediction=[]
for i in range(len(input)):
    traj_rotated=transport_inverse.affine_transform.predict(input[i].reshape(1,-1))
    delta_map_mean, std= transport_inverse.gp_delta_map.predict(traj_rotated)
    transported_traj = traj_rotated + delta_map_mean 
    var_derivative=std[0][:]**2/transport_inverse.gp_delta_map.kernel_params_[0]**2
    sigma_noise_prediction.append(var_derivative)



sigma_noise_prediction=np.array(sigma_noise_prediction)

std_epi=gp_deltaX1.predict(input)[1]

var_epi=std_epi**2


std_hetero= np.sqrt(np.sum(sigma_noise_prediction+var_epi,1))


std_hetero=std_hetero.reshape(X.shape)


# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Y, np.sum(var_epi,1).reshape(X.shape) , linewidth=0, antialiased=True, cmap=plt.cm.inferno)
# plt.axis('off')

# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Y, np.sum(sigma_noise_prediction,1).reshape(X.shape) , linewidth=0, antialiased=True, cmap=plt.cm.inferno)
# plt.axis('off')

ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, std_hetero , linewidth=0, antialiased=True, cmap=plt.cm.inferno)
plt.axis('off')
plt.show()