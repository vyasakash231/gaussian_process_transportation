"""
Authors: Giovanni Franzes, June 2024
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from policy_transportation import AffineTransform
import numpy as np
import Quaternion

class PolicyTransportation():  
    def __init__(self, method):
        super(PolicyTransportation, self).__init__()
        self.delta_map=method

    def fit(self, source_distribution, target_distribution, do_scale=False, do_rotation=True):
        self.affine_transform=AffineTransform(do_scale=do_scale, do_rotation=do_rotation)
        self.affine_transform.fit(source_distribution, target_distribution)

        source_distribution=self.affine_transform.predict(source_distribution)  
 
        self.delta_distribution = target_distribution - source_distribution

        self.delta_map.fit(source_distribution, self.delta_distribution)  

    def transport(self, pos, return_std=True):  # X=state tranform

        pos_rotated=self.affine_transform.predict(pos)  # γ(X=pose)
        if return_std==True:
            delta_map_mean, delta_map_std= self.delta_map.predict(pos_rotated, return_std=return_std)
        else:
            delta_map_mean= self.delta_map.predict(pos_rotated, return_std=return_std)
        pos_transported = pos_rotated + delta_map_mean  # X̂ := Φ(X) = γ(X=pose) + Ψ(γ(X=pose))  

        return pos_transported, delta_map_std
    
    def transport_velocity(self, pos, vel, return_var=True):  # X=state and X_dot=state_dynamics tranform
        pos_rotated=self.affine_transform.predict(pos)
        J_gamma= self.affine_transform.derivative(pos)
        if return_var==True:
            J_psi, J_psi_var=self.delta_map.derivative(pos_rotated, return_var=return_var)
        else:
            J_psi= self.delta_map.derivative(pos_rotated, return_var=return_var)
        
        J_phi= J_gamma + J_psi @ J_gamma
        
        print("Is the map locally diffeomorphic?", np.all(np.abs(np.linalg.det(J_phi)) > 0))

        vel = vel[:,:,np.newaxis]

        vel_rotated=  J_gamma @ vel
        var_vel_transported=J_psi_var @ vel_rotated**2

        vel_transported= J_phi @ vel

        vel_transported=vel_transported[:,:,0]
        var_vel_transported=var_vel_transported[:,:,0]

        return vel_transported, var_vel_transported
        
    def transport_orientation(self, pos, ori):
        J_phi=self.delta_map.derivative(pos)
        J_gamma= self.affine_transform.derivative(pos)
        J_phi= J_gamma + J_phi @ J_gamma

        print("Is the map locally diffeomorphic?", np.all(np.linalg.det(J_phi) > 0))
        if J_phi[0].shape[0]==3:
    
            quat=Quaternion.from_float_array(ori)
            quat_J_phi = Quaternion.from_rotation_matrix(J_phi, nonorthogonal=True)
            quat_transport=quat_J_phi * quat
            ori_transported= Quaternion.as_float_array(quat_transport)
            return ori_transported

        else:
            print("The Jacobain of the map as shape ", J_phi[0].shape, " but it should be (3x3)")
            print("Robot orientation is not transported")


    def sample_transportation(self, pos):
        pos_rotated=self.affine_transform.predict(pos)
        delta_map_samples= self.delta_map.samples(pos_rotated)
        training_traj_samples = pos_rotated + delta_map_samples 
        return training_traj_samples


