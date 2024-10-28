import warnings
from math import *
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

try:
    from .obs_utils import get_directional_weighted_sum  # for global run
except:
    from obs_utils import get_directional_weighted_sum  # for local run
else:
    pass

class ObstacleModulationSystem:
    def __init__(self, obstacles, automatic_reference_point=False):  # if true, this will update the referance point dynamically):
        self.obstacles = obstacles

        # Adjust dynamic center
        if automatic_reference_point:
            self.update_reference_points()

    @staticmethod
    def transform(angle: float) -> np.ndarray:
        if angle == 0:
            return np.eye(2)
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    
    @staticmethod
    def transform_global2relative_dir(obtacle, direction):
        """Transform a direction, velocity or relative position to the obstacle-frame"""
        if obtacle['orientation'] == 0:
            return direction

        R = ObstacleModulationSystem.transform(obtacle['orientation'])
        return R.T @ direction

    @staticmethod
    def single_obstacle_modulation_matrix(zeta: np.ndarray, obstacle: Dict, c: int) -> Tuple[np.ndarray, np.ndarray]:
        obstacle_center = np.array(obstacle['center'])
        reference_point = np.array(obstacle['reference_point'])
        d1, d2 = obstacle['axis_length']
        margin = obstacle['margin']

        if obstacle['orientation'] != 0:
            orientation = np.radians(obstacle['orientation'])
            R = ObstacleModulationSystem.transform(orientation)
            reference_point = R @ reference_point

        reference_point += obstacle_center  # checked
        r = zeta[:2, :] - reference_point[:, np.newaxis]  # checked
        
        if obstacle['shape'] == 'ellipse':
            gamma = ObstacleModulationSystem.get_gamma_ellipse(zeta, obstacle, c)  # checked
            
            norm_of_ref = np.linalg.norm(r, axis=0)
            r_norm = np.where(norm_of_ref != 0, r / norm_of_ref, np.ones_like(r) / 2)  # checked
            
            zeta_ = zeta[:2, :] - obstacle_center[:, np.newaxis]  # checked

            if obstacle['orientation'] != 0:
                orientation = np.radians(obstacle['orientation'])
                R = ObstacleModulationSystem.transform(orientation)
                zeta_ = R.T @ zeta_

            p = np.array([1, 1])
            if margin == 0:
                gx = (2 * p[0] / d1**(2*p[0])) * zeta_[0, :]**(2*p[0]-1)
                gy = (2 * p[1] / d2**(2*p[1])) * zeta_[1, :]**(2*p[1]-1)
            else:
                d1 += 2 * margin
                d2 += 2 * margin
                gx = (2 * p[0] / d1**(2*p[0])) * zeta_[0, :]**(2*p[0]-1)
                gy = (2 * p[1] / d2**(2*p[1])) * zeta_[1, :]**(2*p[1]-1)

            n_vec = np.vstack((gx, gy))  # checked

            normal_norm = np.linalg.norm(n_vec, axis=0)
            nonzero_norms = normal_norm != 0
            n_vec[:, nonzero_norms] /= normal_norm[nonzero_norms]
            n_vec[0, ~nonzero_norms] = 1

            if obstacle['orientation'] != 0:
                orientation = np.radians(obstacle['orientation'])
                R = ObstacleModulationSystem.transform(orientation)
                n_vec = R @ n_vec  # checked
        
        elif obstacle['shape'] == 'cuboid':
            gamma = ObstacleModulationSystem.get_gamma_cuboid(zeta, obstacle, c)

            norm_of_ref = np.linalg.norm(r, axis=0)
            r_norm = np.where(norm_of_ref != 0, r / norm_of_ref, np.ones_like(r) / 2)

            zeta_ = zeta[:2, :] - obstacle_center[:, np.newaxis]
            semi_axis = np.array([d1/2, d2/2])[:, np.newaxis]

            if obstacle['orientation'] != 0:
                orientation = np.radians(obstacle['orientation'])
                R = ObstacleModulationSystem.transform(orientation)
                zeta_ = R.T @ zeta_

            ind_relevant = np.abs(zeta_) > semi_axis

            for i in range(c):
                if not np.any(ind_relevant[:, i]):
                    minimum_factor = np.max(np.abs(zeta[:, i]) / semi_axis.ravel())
                    zeta[:, i] /= minimum_factor**2
                    ind_relevant[:, i] = np.abs(zeta[:, i]) > semi_axis.ravel()

            n_vec = np.zeros_like(zeta_)
            n_vec[0, ind_relevant[0, :]] = zeta_[0, ind_relevant[0, :]] - (semi_axis[0] * np.sign(zeta_[0, ind_relevant[0, :]]))
            n_vec[1, ind_relevant[1, :]] = zeta_[1, ind_relevant[1, :]] - (semi_axis[1] * np.sign(zeta_[1, ind_relevant[1, :]]))

            normal_norm = np.linalg.norm(n_vec, axis=0)
            nonzero_norms = normal_norm != 0
            n_vec[:, nonzero_norms] /= normal_norm[nonzero_norms]
            n_vec[0, ~nonzero_norms] = 1

            if obstacle['orientation'] != 0:
                orientation = np.radians(obstacle['orientation'])
                R = ObstacleModulationSystem.transform(orientation)
                n_vec = R @ n_vec

        z_basis_3D = np.tile(np.array([0, 0, 1])[:, np.newaxis], (1, c))
        e = np.cross(np.vstack((n_vec, np.zeros((1, c)))), z_basis_3D, axis=0)
        
        E_ortho = np.zeros((2, 2, c))
        E_ortho[:, 1, :] = e[:2, :]   # checked
        E_ortho[:, 0, :] = n_vec  # checked

        E = np.copy((E_ortho))
        E[:, 0, :] = r_norm   # checked
        return E, E_ortho, gamma 

    @staticmethod
    def get_gamma_ellipse(zeta: np.ndarray, obstacle: Dict, c: int) -> np.ndarray:
        zeta = zeta[:2, :] - obstacle['center'][:, np.newaxis]
        if obstacle['orientation'] != 0:
            orientation = np.radians(obstacle['orientation'])
            R = ObstacleModulationSystem.transform(orientation)
            zeta = R.T @ zeta
        surface_point = ObstacleModulationSystem.get_point_on_surface(zeta, obstacle, c)

        distance_surface = np.linalg.norm(surface_point, axis=0)
        distance_zeta = np.linalg.norm(zeta, axis=0)

        distance = np.where(distance_zeta > distance_surface, np.linalg.norm(zeta - surface_point, axis=0), distance_zeta / distance_surface - 1)
        return distance + 1

    @staticmethod
    def get_point_on_surface(zeta: np.ndarray, obstacle: Dict, c: int) -> np.ndarray:
        margin = obstacle['margin']
        semi_axis_length = np.array(obstacle['axis_length']) / 2

        if margin == 0:
            circle_position = zeta / semi_axis_length[:, np.newaxis]
        else:
            circle_position = zeta / (semi_axis_length[:, np.newaxis] + margin)

        pos_norm = np.linalg.norm(circle_position, axis=0)
        surface_point = np.zeros_like(zeta)

        zero_norm = pos_norm == 0
        surface_point[:, zero_norm] = np.array([semi_axis_length[0] + margin if margin != 0 else 0, 0])[:, np.newaxis]
        surface_point[:, ~zero_norm] = zeta[:, ~zero_norm] / pos_norm[~zero_norm]
        return surface_point

    @staticmethod
    def get_gamma_cuboid(zeta: np.ndarray, obstacle: Dict, c: int) -> np.ndarray:
        boundary_power_factor = 1
        gamma = np.zeros(c)

        for i in range(c):
            distance_surface = ObstacleModulationSystem.get_distance_to_surface(zeta[:, i], obstacle)
            if distance_surface < 0:
                gamma[i] = (np.linalg.norm(zeta[:, i]) / (np.linalg.norm(zeta[:, i]) - distance_surface)) ** boundary_power_factor
            else:
                gamma[i] = distance_surface + 1
        return gamma

    @staticmethod
    def get_distance_to_surface(zeta: np.ndarray, obstacle: Dict) -> float:
        zeta = zeta[:2] - obstacle['center']
        if obstacle['orientation'] != 0:
            orientation = np.radians(obstacle['orientation'])
            R = ObstacleModulationSystem.transform(orientation)
            zeta = R.T @ zeta

        margin = obstacle['margin']
        relative_zeta = np.abs(zeta) - (np.array(obstacle['axis_length']) / 2)

        if np.any(relative_zeta > 0):
            relative_zeta = np.maximum(relative_zeta, 0)
            distance = np.linalg.norm(relative_zeta)
            if distance > margin:
                return distance - margin
            return margin - distance
        else:
            distance = margin + (-1) * np.max(relative_zeta)

        return (-1) * (distance / (np.linalg.norm(zeta) + distance))

    @staticmethod
    def omega_denominator(d_store: np.ndarray) -> np.ndarray:
        no_of_obstacle, no_of_agent = d_store.shape
        denom = np.zeros(no_of_agent)
        one_mat = np.ones(no_of_obstacle - 1)

        for i in range(no_of_obstacle):
            idx = np.arange(no_of_obstacle) != i
            denom += np.prod(d_store[idx, :] - one_mat[:, np.newaxis], axis=0)
        return denom

    def mutiple_obstacle_modulation_matrix(self, state: np.ndarray, c: int, initial_velocity: np.ndarray) -> np.ndarray:
        M_combined = np.stack([np.eye((2))] * c, axis=2)

        for k, obs_k in enumerate(self.obstacles):
            E_k, E_ortho_k, gamma_k = ObstacleModulationSystem.single_obstacle_modulation_matrix(state, obs_k, c)   # checked
            
            gamma_store = np.zeros((len(self.obstacles), len(gamma_k)))
            numerator = np.ones_like(gamma_k)
            
            for i, obs_i in enumerate(self.obstacles):
                if i != k:
                    _, _, gamma_i = ObstacleModulationSystem.single_obstacle_modulation_matrix(state, obs_i, c)
                    
                    numerator = numerator * (gamma_i - 1)
                    gamma_store[i, :] = gamma_i
                else:
                    gamma_store[i, :] = gamma_k

            denominator = ObstacleModulationSystem.omega_denominator(gamma_store)
            omega_k = numerator / denominator    # checked
            
            # Compute eigenvalues
            lambda_1 = 1 - omega_k / gamma_k
            lambda_2 = 1 + omega_k / gamma_k
    
            for i in range(c):
                D_k = np.diag([lambda_1[i], lambda_2[i]])
                M_k = E_k[:, :, i] @ D_k @ np.linalg.inv(E_k[:, :, i])
                M_combined[:, :, i] = M_combined[:, :, i] @ M_k     
        return M_combined
    

    @staticmethod
    def get_relative_obstacle_velocity(
        position: np.ndarray,
        obstacle_list,
        E_orth: np.ndarray,
        weights: list,
        ind_obstacles: Optional[int] = None,
        gamma_list: Optional[list] = None,
        cut_off_gamma: float = 1e4,
        velocity_only_in_positive_normal_direction: bool = True,
        normal_weight_factor: float = 1.3) -> np.ndarray:
        """Get the relative obstacle velocity"""
        n_obstacles = len(obstacle_list)

        if gamma_list is None:
            gamma_list = np.zeros(n_obstacles)

            for n, obstacle in enumerate(obstacle_list):
                if obstacle['shape'] == 'ellipse':
                    gamma_list[n] = ObstacleModulationSystem.get_gamma_ellipse(position, obstacle, 1)  # checked

                if obstacle['shape'] == 'cuboid':
                    gamma_list[n] = ObstacleModulationSystem.get_gamma_ellipse(position, obstacle, 1)  # checked

        if ind_obstacles is None:
            ind_obstacles = gamma_list < cut_off_gamma
            gamma_list = gamma_list[ind_obstacles]

        obs = obstacle_list
        ind_obs = ind_obstacles
        dim = position.shape[0]

        xd_obs = np.zeros((dim))
        for ii, it_obs in zip(range(np.sum(ind_obs)), np.arange(n_obstacles)[ind_obs]):
            if obs[it_obs]['angular_velocity'] is None:
                xd_w = np.zeros(dim)
            else:
                xd_w = np.cross(np.hstack(([0, 0], obs[it_obs]['angular_velocity'])),
                    np.hstack((position - np.array(obs[it_obs]['center']), 0)))
                xd_w = xd_w[0:2]

            weight_angular = np.exp(-1.0 * (np.max([gamma_list[ii], 1]) - 1))

            linear_velocity = obs[it_obs]['linear_velocity']

            if velocity_only_in_positive_normal_direction:
                lin_vel_local = (E_orth[:, :, ii]).T.dot(obs[it_obs]['linear_velocity'])
                if lin_vel_local[0] < 0:
                    # Obstacle is moving towards the agent
                    linear_velocity = np.zeros(lin_vel_local.shape[0])
                else:
                    # For safety in close region, we multiply the velocity
                    lin_vel_local[0] = normal_weight_factor * lin_vel_local[0]
                    linear_velocity = E_orth[:, 0, ii].dot(lin_vel_local[0])

                weight_linear = np.exp(-1 / 1 * (np.max([gamma_list[ii], 1]) - 1))
            xd_obs_n = weight_linear * linear_velocity + weight_angular * xd_w

            xd_obs = xd_obs + xd_obs_n * weights[ii]
        return xd_obs


    def obs_avoidance_interpolation_moving(
            self, state: np.ndarray, c: int, 
            initial_velocity: np.ndarray, 
            cut_off_gamma=1e6,
            evaluate_in_global_frame=True):

        D = np.zeros((2,2,len(self.obstacles),c))
        E = np.zeros((2,2,len(self.obstacles),c))
        E_ortho = np.zeros((2,2,len(self.obstacles),c))
        E_inv = np.zeros((2,2,len(self.obstacles),c))
        omega = np.zeros((len(self.obstacles), c))
        gamma = np.zeros((len(self.obstacles), c))

        for k, obs_k in enumerate(self.obstacles):
            E_k, E_ortho_k, gamma_k = ObstacleModulationSystem.single_obstacle_modulation_matrix(state, obs_k, c)   # checked
            
            gamma_store = np.zeros((len(self.obstacles), len(gamma_k)))
            numerator = np.ones_like(gamma_k)
            
            for i, obs_i in enumerate(self.obstacles):
                if i != k:
                    _, _, gamma_i = ObstacleModulationSystem.single_obstacle_modulation_matrix(state, obs_i, c)
                    
                    numerator = numerator * (gamma_i - 1)
                    gamma_store[i, :] = gamma_i
                else:
                    gamma_store[i, :] = gamma_k

            denominator = ObstacleModulationSystem.omega_denominator(gamma_store)
            omega_k = numerator / denominator    # checked
            
            # Compute eigenvalues
            lambda_1 = 1 - 1 / gamma_k
            lambda_2 = 1 + 1 / gamma_k
    
            # Create the diagonal matrices for each set of eigenvalues
            D_k = np.stack([np.diag([l1, l2]) for l1, l2 in zip(lambda_1, lambda_2)], axis=2)

            omega[k, :] = omega_k
            gamma[k, :] = gamma_k
            
            D[:,:,k,:] = D_k
            E[:,:,k,:] = E_k
            E_ortho[:,:,k,:] = E_ortho_k                

            E_inv[:,:,k,:] = np.linalg.inv(E_k.transpose(2, 0, 1)).transpose(1, 2, 0)

        final_velocity = np.zeros(initial_velocity.shape)
        for agent in range(c):
            D_obs = D[:,:,:,agent]
            E_obs = E[:,:,:,agent]
            E_obs_ortho = E_ortho[:,:,:,agent]
            E_obs_inv = E_inv[:,:,:,agent]

            omega_obs = omega[:,agent]

            ind_obs = gamma[:,agent] < cut_off_gamma
            
            xd_obs = ObstacleModulationSystem.get_relative_obstacle_velocity(position=state[:,agent],
                                                                    obstacle_list=self.obstacles,
                                                                    E_orth=E_obs_ortho,
                                                                    gamma_list=gamma[:,agent],
                                                                    weights=omega_obs)

            # Computing the relative velocity with respect to the obstacle
            relative_velocity = initial_velocity[:,agent] - xd_obs

            rel_velocity_norm = np.linalg.norm(relative_velocity)
            if rel_velocity_norm:
                rel_velocity_normalized = relative_velocity / rel_velocity_norm
            else:
                # Zero velocity
                return xd_obs

            # Keep either way, since avoidance from attractor might be needed
            relative_velocity_hat = np.zeros((2, len(self.obstacles)))    # here dim = 2 
            relative_velocity_hat_magnitude = np.zeros((len(self.obstacles)))

            n = 0
            for n in np.arange(len(self.obstacles))[ind_obs]:
                if self.obstacles[n]['repulsion_coeff'] > 1 and E_obs_ortho[:, 0, n].T.dot(relative_velocity) < 0:
                    # Only consider boundary when moving towards (normal direction)
                    # OR if the object has positive repulsion-coefficient (only consider it at front)
                    relative_velocity_hat[:, n] = relative_velocity

                else:
                    # Matrix inversion cost between O(n^2.373) - O(n^3)
                    if not evaluate_in_global_frame:
                        relative_velocity_temp = ObstacleModulationSystem.transform_global2relative_dir(self.obstacles[n], relative_velocity)
                    else:
                        relative_velocity_temp = np.copy(relative_velocity)

                    # Modulation with M = E @ D @ E^-1
                    relative_velocity_trafo = E_obs_inv[:, :, n].dot(relative_velocity_temp)  # E^-1 @ f(X)
                    stretched_velocity = D_obs[:, :, n].dot(relative_velocity_trafo)  # D @ E^-1 @ f(X)

                    if D_obs[0, 0, n] < 0:
                        # Repulsion in tangent direction, too, have really active repulsion
                        factor_tangent_repulsion = 2
                        tang_vel_norm = np.linalg.norm(relative_velocity_trafo[1:])
                        stretched_velocity[0] += ((-1) * D_obs[0, 0, n] * tang_vel_norm * factor_tangent_repulsion)

                    relative_velocity_hat[:, n] = E_obs[:, :, n].dot(stretched_velocity)  # E @ D @ E^-1 @ f(X)

                relative_velocity_hat_magnitude[n] = np.sqrt(np.sum(relative_velocity_hat[:, n] ** 2))

            relative_velocity_hat_normalized = np.zeros(relative_velocity_hat.shape)
            ind_nonzero = relative_velocity_hat_magnitude > 0
            if np.sum(ind_nonzero):
                relative_velocity_hat_normalized[:, ind_nonzero] = relative_velocity_hat[:, ind_nonzero] / np.tile(relative_velocity_hat_magnitude[ind_nonzero], (2, 1))  # here dim = 2 

            if rel_velocity_norm:
                weighted_direction = get_directional_weighted_sum(
                    null_direction=rel_velocity_normalized,
                    directions=relative_velocity_hat_normalized,
                    weights=omega_obs)

            else:
                weighted_direction = np.sum(
                    np.tile(omega_obs, (1, relative_velocity_hat_normalized.shape[0])).T
                    * relative_velocity_hat_normalized, axis=0)

            relative_velocity_magnitude = np.sum(relative_velocity_hat_magnitude * omega_obs)
            vel_final = relative_velocity_magnitude * weighted_direction.squeeze()

            vel_final = vel_final + xd_obs

            final_velocity[:,agent] = vel_final
        return final_velocity


    def plot_multiple_obstacles(self, plot_limits: List[float] = None):
        if plot_limits is None:
            plot_limits = [-50, 50, -50, 50]

        x, y = np.meshgrid(np.linspace(plot_limits[0], plot_limits[1], 500),
                           np.linspace(plot_limits[2], plot_limits[3], 500))

        for i, obstacle in enumerate(self.obstacles):
            xc, yc = obstacle['center']
            reference_point = np.array(obstacle['reference_point'])
            semi_axis_length = np.array(obstacle['axis_length']) / 2

            m = 10 if obstacle['shape'] == 'cuboid' else 2

            x_rot = x - xc
            y_rot = y - yc

            if obstacle['orientation'] != 0:
                rotated_points = np.vstack((x_rot.ravel(), y_rot.ravel()))
                orientation = np.radians(obstacle['orientation'])
                R = ObstacleModulationSystem.transform(orientation)
                rotated_points = R.T @ rotated_points
                x_rot = rotated_points[0, :].reshape(x.shape)
                y_rot = rotated_points[1, :].reshape(y.shape)

            F = (x_rot / semi_axis_length[0])**m + (y_rot / semi_axis_length[1])**m - 1

            plt.contourf(x, y, F, levels=[-np.inf, 0], colors=[obstacle['color']], alpha=0.85)
            plt.contour(x, y, F, levels=[0], colors="k", linewidths=0.8)

            plt.plot(xc, yc, 'ko', markersize=5, markerfacecolor='k')

            if obstacle['orientation'] != 0:
                orientation = np.radians(obstacle['orientation'])
                R = ObstacleModulationSystem.transform(orientation)
                reference_point = R @ reference_point

            plt.plot(reference_point[0] + xc, reference_point[1] + yc, 'r+', markersize=8, markeredgewidth=2)

        # plt.xlim(plot_limits[0], plot_limits[1])
        # plt.ylim(plot_limits[2], plot_limits[3])
        # plt.gca().set_aspect('equal', adjustable='box')
