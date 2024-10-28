import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from obstacle_avoidance_Linear_DS import ObstacleModulationSystem

# Time settings
dt = 0.01
T = np.arange(0, 10 + dt, dt)

# Goal state
x_g, y_g = 35.0, 25.0

# Define obstacles
obstacles: List[Dict] = [
    {'shape': 'ellipse', 'center': np.array([22.0, 22.5]), 'reference_point': np.array([0.0, 3.0]), 'axis_length': np.array([2.0, 6.0]), 'orientation': 0, 'margin': 0, 'repulsion_coeff': 1.0, 'linear_velocity': np.array([0.0, 0.0]), 'angular_velocity': None, 'color': [0.5, 0.2, 0.2]},
    {'shape': 'ellipse', 'center': np.array([24.5, 26.5]), 'reference_point': np.array([0.0, 3.0]), 'axis_length': np.array([2.0, 6.0]), 'orientation': 120, 'margin': 0, 'repulsion_coeff': 1.0, 'linear_velocity': np.array([0.0, 0.0]), 'angular_velocity': None, 'color': [0.5, 0.2, 0.2]},
    {'shape': 'ellipse', 'center': np.array([19.5, 26.5]), 'reference_point': np.array([0.0, 3.0]), 'axis_length': np.array([2.0, 6.0]), 'orientation': 240, 'margin': 0, 'repulsion_coeff': 1.0, 'linear_velocity': np.array([0.0, 0.0]), 'angular_velocity': None, 'color': [0.5, 0.2, 0.2]},

    {'shape': 'ellipse', 'center': np.array([8.0, 12.0]), 'reference_point': np.array([0.0, 0.0]), 'axis_length': np.array([2.0, 7.0]), 'orientation': 0, 'margin': 0, 'repulsion_coeff': 1.0, 'linear_velocity': np.array([0.0, 0.0]), 'angular_velocity': None, 'color': [0.5, 0.2, 0.2]},
    {'shape': 'ellipse', 'center': np.array([8.0, 12.0]), 'reference_point': np.array([0.0, 0.0]), 'axis_length': np.array([7.0, 2.0]), 'orientation': 0, 'margin': 0, 'repulsion_coeff': 1.0, 'linear_velocity': np.array([0.0, 0.0]), 'angular_velocity': None, 'color': [0.5, 0.2, 0.2]},

    {'shape': 'ellipse', 'center': np.array([10.5, 20.0]), 'reference_point': np.array([0.0, 2.8]), 'axis_length': np.array([2.0, 6.0]), 'orientation': 300, 'margin': 0, 'repulsion_coeff': 1.0, 'linear_velocity': np.array([0.0, 0.0]), 'angular_velocity': None, 'color': [0.5, 0.2, 0.2]},
    {'shape': 'ellipse', 'center': np.array([10.5, 22.5]), 'reference_point': np.array([0.0, 2.8]), 'axis_length': np.array([2.0, 6.0]), 'orientation': 240, 'margin': 0, 'repulsion_coeff': 1.0, 'linear_velocity': np.array([0.0, 0.0]), 'angular_velocity': None, 'color': [0.5, 0.2, 0.2]},

    # {'shape': 'cuboid', 'center': np.array([10.5, 34.0]), 'reference_point': np.array([0.0, 0.0]), 'axis_length': np.array([2.0, 6.0]), 'orientation': 0, 'margin': 0, 'repulsion_coeff': 1.0, 'linear_velocity': np.array([0.0, 0.0]), 'angular_velocity': None, 'color': [0.5, 0.2, 0.2]},
    # {'shape': 'cuboid', 'center': np.array([10.5, 34.0]), 'reference_point': np.array([0.0, 0.0]), 'axis_length': np.array([2.0, 6.0]), 'orientation': 90, 'margin': 0, 'repulsion_coeff': 1.0, 'linear_velocity': np.array([0.0, 0.0]), 'angular_velocity': None, 'color': [0.5, 0.2, 0.2]},

    # {'shape': 'cuboid', 'center': np.array([10.0, 22.0]), 'reference_point': np.array([0.0, 0.0]), 'axis_length': np.array([2.0, 6.0]), 'orientation': 0, 'margin': 0},
    # {'shape': 'cuboid', 'center': np.array([6.75, 22.0]), 'reference_point': np.array([0.0, -2.6]), 'axis_length': np.array([2.0, 6.0]), 'orientation': 90, 'margin': 0}
]


# Initial conditions
no_of_agents = 50
Zeta_vec = np.zeros((2, no_of_agents, len(T)))
Zeta_vec[1, :, 0] = np.linspace(0, 50, no_of_agents)

# Create an instance of ObstacleModulationSystem
obs = ObstacleModulationSystem(obstacles, automatic_reference_point=False)  # "automatic_reference_point" is not working yet


# # My Method
# for i in range(len(T) - 1):
#     Zeta = Zeta_vec[:, :, i]
   
#     # Original dynamical system
#     f = np.array([x_g - Zeta[0, :], y_g - Zeta[1, :]])
    
#     # Modulation
#     M = obs.mutiple_obstacle_modulation_matrix(Zeta, no_of_agents, f)

#     # Modified dynamical system
#     Zeta_dot_vec = np.zeros_like(f)
#     for k in range(no_of_agents):
#         Zeta_dot_vec[:, k] = M[:, :, k] @ f[:, k]
    
#     # Update state
#     Zeta_vec[:, :, i+1] = Zeta + Zeta_dot_vec * dt


#  LukasHuber Code
for i in range(len(T) - 1):    
    Zeta = Zeta_vec[:, :, i]

     # Original dynamical system
    f = np.array([x_g - Zeta[0, :], y_g - Zeta[1, :]])

    # Modulation dynamical system
    Zeta_dot_vec = obs.obs_avoidance_interpolation_moving(Zeta, no_of_agents, f)
   
    # Update state
    Zeta_vec[:, :, i+1] = Zeta + Zeta_dot_vec * dt
    

# Plotting
plt.figure(figsize=(12, 10))

# Plot obstacles
obs.plot_multiple_obstacles([0, 40, 0, 40])

# Plot trajectories
for j in range(no_of_agents):
    plt.plot(Zeta_vec[0, j, :], Zeta_vec[1, j, :], linewidth=1, color=[0.45, 0.45, 0.45])

# Plot start and goal points
plt.plot(Zeta_vec[0, :, 0], Zeta_vec[1, :, 0], 'go', markersize=4, label='Start')
plt.plot(x_g, y_g, 'k*', markersize=10, label='Goal')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Agent Trajectories with Obstacle Modulation')
plt.legend()
plt.show()