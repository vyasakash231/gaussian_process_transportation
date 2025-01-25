import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from obstacle_avoidance_Linear_DS import ObstacleModulationSystem

def rotation_matrix_2d(angle):
    """Returns a 2D rotation matrix for given angle."""
    return np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])

def wavy_dynamics(pos, attractor):
    """Computes the dynamical system at point (x,y)."""
    if pos.ndim == 2:
        pos = pos.reshape(-1)

    diff = attractor - pos
    dist = np.linalg.norm(diff)
    
    # Create rotation based on distance
    R = rotation_matrix_2d(np.sin(dist))
    
    # Compute velocity
    velocity = R @ diff
    return velocity.reshape((2,1))

## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##

# Define obstacles
obstacles: List[Dict] = [
    {'shape': 'ellipse', 'center': np.array([0.20, -3.1]), 'reference_point': np.array([0.0, 0.3]), 'axis_length': np.array([0.3, 0.7]), 'orientation': 0, 'margin': 0, 'repulsion_coeff': 1.0, 'linear_velocity': np.array([0.0, 0.0]), 'angular_velocity': None, 'color': [0.5, 0.2, 0.2]},
    {'shape': 'ellipse', 'center': np.array([0.45, -2.65]), 'reference_point': np.array([0.0, 0.3]), 'axis_length': np.array([0.3, 0.7]), 'orientation': 120, 'margin': 0, 'repulsion_coeff': 1.0, 'linear_velocity': np.array([0.0, 0.0]), 'angular_velocity': None, 'color': [0.5, 0.2, 0.2]},
    {'shape': 'ellipse', 'center': np.array([-0.05, -2.65]), 'reference_point': np.array([0.0, 0.3]), 'axis_length': np.array([0.3, 0.7]), 'orientation': 240, 'margin': 0, 'repulsion_coeff': 1.0, 'linear_velocity': np.array([0.0, 0.0]), 'angular_velocity': None, 'color': [0.5, 0.2, 0.2]},
    ]

# Create an instance of ObstacleModulationSystem
obs = ObstacleModulationSystem(obstacles, automatic_reference_point=False)  # "automatic_reference_point" is not working yet

# Set up the grid
x_min, x_max = -5, 1
y_min, y_max = -5, 1
n_points = 20

x = np.linspace(x_min, x_max, n_points)
y = np.linspace(y_min, y_max, n_points)
X, Y = np.meshgrid(x, y)

# Attractor position
attractor = np.array([-1, -1])

# Compute vector field
U = np.zeros_like(X)
V = np.zeros_like(Y)
Mod_U = np.zeros_like(X)
Mod_V = np.zeros_like(Y)

for i in range(n_points):
    for j in range(n_points):
        zeta = np.array([[X[i,j]], [Y[i,j]]])

        velocity = wavy_dynamics(zeta, attractor)
        U[i,j], V[i,j] = velocity[0,0], velocity[1,0]

        # Modulation dynamical system
        Mod_velocity = obs.obs_avoidance_interpolation_moving(zeta, 1, velocity)
        Mod_U[i,j], Mod_V[i,j] = Mod_velocity[0], Mod_velocity[1]

        # # # Modulation dynamical system
        weights = obs.normalized_weights(zeta)

        ref_velocity = np.zeros((2, len(obstacles)))
        for idx, obstacle in enumerate(obstacles):
            xr, yr = obstacle["reference_point"]
            ref_velocity[:, [idx]] = wavy_dynamics(np.array([[xr],[yr]]), attractor)

        # Rotational Average
        # velocity_1 = np.sum(weights[:, np.newaxis] * ref_velocity, axis=0)
        
        # convergence dynamics
        


# Create two subplots: one with normalized and one with unnormalized vectors
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Normalized vectors
norm = np.sqrt(U**2 + V**2)
U_norm = U / norm 
V_norm = V / norm

Mod_norm = np.sqrt(Mod_U**2 + Mod_V**2)
Mod_U_norm = Mod_U / Mod_norm 
Mod_V_norm = Mod_V / Mod_norm

# Plot 1
plt.sca(ax[0])  # Set the current axes for obstacle plotting
# Q1 = ax[0].quiver(X, Y, U_norm, V_norm, norm, cmap='viridis', scale=35, width=0.005)
# plt.colorbar(Q1, ax=ax[0], label='Vector Magnitude')
ax[0].quiver(X, Y, U_norm, V_norm, scale=35, width=0.005, color='g')
ax[0].quiver(X, Y, Mod_U_norm, Mod_V_norm, scale=35, width=0.005, color='m')
ax[0].plot(attractor[0], attractor[1], 'k*', markersize=15, label='Attractor')
obs.plot_multiple_obstacles([x_min, x_max, y_min, y_max])  # Plot obstacles
ax[0].set_title('Normalized Vector Field')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].legend()
ax[0].set_aspect('equal')


# Plot 2
plt.sca(ax[1])  # Set the current axes for obstacle plotting
ax[1].streamplot(X, Y, U_norm, V_norm, density=1.5, color='b', linewidth=1.5)  # Plot streamlines
ax[1].plot(attractor[0], attractor[1], 'k*', markersize=15, label='Attractor')  # Plot attractor
# obs.plot_multiple_obstacles([x_min, x_max, y_min, y_max])  # Plot obstacles
ax[1].set_title('Normalized Streamlines')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].legend()
ax[1].set_aspect('equal')

plt.show()