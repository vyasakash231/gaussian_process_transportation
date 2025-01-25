import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d, LinearNDInterpolator, NearestNDInterpolator
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from policy_transportation import GaussianProcess 


"""hyperspherical surface projector for star-shaped domains"""
class HypersphericalProjector:
    def __init__(self, boundary_points, center=None):
        self.boundary_points = np.array(boundary_points)
        
        if center is None:
            self.center = np.mean(boundary_points, axis=0)
        else:
            self.center = np.array(center)
            
        # Convert boundary points to spherical coordinates
        self.boundary_spherical = self._to_hyp_spherical(self.boundary_points - self.center)
        
        # Create interpolator for radial distance
        self._setup_interpolators()
        
    def _to_hyp_spherical(self, points):
        """Convert Cartesian (x,y,z) to spherical (r,θ,φ) coordinates."""
        if len(points.shape) == 1:
            points = points.reshape(1, -1)

        if len(self.center) == 2:
            x, y = points[:, 0], points[:, 1]
            
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            
            # Ensure theta is in [0, 2π]
            theta = np.where(theta < 0, theta + 2*np.pi, theta)
            return np.column_stack([r, theta]) 

        if len(self.center) == 3: 
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            
            r = np.sqrt(x**2 + y**2 + z**2)
            # Handle points at or very near the center
            mask = r > 1e-10
            
            theta = np.zeros_like(r)
            phi = np.zeros_like(r)
            
            # Only compute angles for non-zero radius points
            theta[mask] = np.arccos(np.clip(z[mask] / r[mask], -1.0, 1.0))
            phi[mask] = np.arctan2(y[mask], x[mask])
            
            # Ensure phi is in [0, 2π]
            phi = np.where(phi < 0, phi + 2*np.pi, phi)
            return np.column_stack([r, theta, phi])
    
    def _to_cartesian(self, hyp_spherical):
        """Convert spherical (r,θ,φ) to Cartesian (x,y,z) coordinates."""
        if len(hyp_spherical.shape) == 1:
            hyp_spherical = hyp_spherical.reshape(1, -1)

        if len(self.center) == 2:
            r, theta = hyp_spherical[:, 0], hyp_spherical[:, 1]
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return np.column_stack([x, y])
        
        if len(self.center) == 3:
            r, theta, phi = hyp_spherical[:, 0], hyp_spherical[:, 1], hyp_spherical[:, 2]
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            return np.column_stack([x, y, z])
    
    def _setup_interpolators(self):
        """Create interpolators for radial distance as function of angles."""
        if len(self.center) == 2:
            angles = self.boundary_spherical[:, 1]
            radii = self.boundary_spherical[:, 0]
        
            # Add periodic points for interpolation
            angles = np.concatenate([angles, [2*np.pi]])
            radii = np.concatenate([radii, [radii[0]]])
            
            # Create periodic interpolator
            self.radius_interpolator = interp1d(angles, radii, kind='linear', bounds_error=False, fill_value=(radii[-1], radii[0]))
        
        if len(self.center) == 3:
            # Extract angles and radii from boundary points
            angles = self.boundary_spherical[:, 1:]  # theta, phi
            radii = self.boundary_spherical[:, 0]

            # Remove any points with zero radius
            valid_mask = radii > 1e-10
            valid_angles = angles[valid_mask]
            valid_radii = radii[valid_mask]
            
            # Create both linear and nearest interpolators
            self.linear_interpolator = LinearNDInterpolator(valid_angles, valid_radii)
            self.nearest_interpolator = NearestNDInterpolator(valid_angles, valid_radii)
    
    """Project points radially onto the boundary surface using spherical coordinates."""
    def project_points(self, points):
        points = np.array(points)
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
            
        # Convert to spherical coordinates relative to center
        points_centered = points - self.center
        points_spherical = self._to_hyp_spherical(points_centered)
        
        # Get angles (theta, phi)
        angles = points_spherical[:, 1:]
        
        if len(self.center) == 2:
            projected_radii = self.radius_interpolator(angles)
        if len(self.center) == 3:
            # Try linear interpolation first
            projected_radii = self.linear_interpolator(angles)
            
            # Use nearest neighbor interpolation for points where linear interpolation failed
            mask = np.isnan(projected_radii)
            if np.any(mask):
                projected_radii[mask] = self.nearest_interpolator(angles[mask])
        
        # Create projected spherical coordinates
        projected_polar = np.column_stack([projected_radii, angles])
        
        # Convert back to Cartesian coordinates
        projected_cartesian = self._to_cartesian(projected_polar)
        projected_points = projected_cartesian + self.center
        
        # Calculate distances
        distances = np.linalg.norm(projected_points - self.center, axis=1)   # Distances from center to projected points
        return projected_points, distances
    

"""Signed Distance Field"""
class SDFCalculator:
    def __init__(self, boundary_points):
        self.boundary_points = boundary_points
        self.segments = np.array([
            [self.boundary_points[i], self.boundary_points[(i + 1) % len(boundary_points)]]
            for i in range(len(boundary_points))])
        
    def signed_distance(self, points):
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
            
        # Calculate distance to each segment and take minimum
        distances = np.inf * np.ones(len(points))
        
        for segment in self.segments:
            p1, p2 = segment
            segment_vec = p2 - p1
            length_sq = np.dot(segment_vec, segment_vec)
            
            if length_sq == 0:
                # Handle degenerate segments
                segment_distances = np.linalg.norm(points - p1, axis=1)
            else:
                # Calculate projection parameters
                t = np.clip(np.dot(points - p1, segment_vec) / length_sq, 0, 1)
                # Calculate projected points
                proj = p1 + np.outer(t, segment_vec)
                segment_distances = np.linalg.norm(points - proj, axis=1)
            
            distances = np.minimum(distances, segment_distances)
        
        # Determine inside/outside using winding number
        signs = self._calculate_signs(points)
        return distances * signs
    
    def calculate_gradient(self, points, epsilon=1e-6):
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
            
        gradients = np.zeros_like(points)
        
        # Calculate gradient components using central differences
        for i in range(points.shape[1]):
            offset = np.zeros_like(points)
            offset[:, i] = epsilon
            
            right = self.signed_distance(points + offset)
            left = self.signed_distance(points - offset)
            
            gradients[:, i] = (right - left) / (2 * epsilon)
            
        # Normalize gradients
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        mask = (norms > 1e-10).flatten()
        gradients[mask] = gradients[mask] / norms[mask]
            
        return gradients
    
    def _calculate_signs(self, points):
        def angle(p, p1, p2):
            v1 = p1 - p
            v2 = p2 - p
            dot = np.sum(v1 * v2, axis=1)
            det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            return np.arctan2(det, dot)
        
        winding_numbers = np.zeros(len(points))
        
        for i in range(len(self.boundary_points)):
            p1 = self.boundary_points[i]
            p2 = self.boundary_points[(i + 1) % len(self.boundary_points)]
            winding_numbers += angle(points, p1, p2)
            
        # Point is inside if winding number is approximately 2π or -2π
        inside = np.abs(np.abs(winding_numbers) - 2 * np.pi) < 0.1
        return np.where(inside, -1., 1.)


class RadialSurfaceProjector:
    def __init__(self, boundary_points, center=None):
        self.boundary_points = boundary_points
        self.center = center
    
    def _get_point_angle(self, point):
        """Calculate angle of point relative to center."""
        vector = point - self.center
        angle = np.arctan2(vector[1], vector[0])
        return angle if angle >= 0 else angle + 2*np.pi
    
    def project_points(self, points):
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
            
        projected = np.zeros_like(points)
        distances = np.zeros(len(points))
        
        for i, point in enumerate(points):
            # Get angle of point relative to center
            angle = self._get_point_angle(point)
            
            # Find intersection with boundary segments
            min_dist = float('inf')
            best_intersection = None
            
            # Create ray direction from angle
            ray_direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Check each boundary segment
            for j in range(len(self.boundary_points)):
                p1 = self.boundary_points[j]
                p2 = self.boundary_points[(j + 1) % len(self.boundary_points)]
                
                # Calculate intersection
                segment = p2 - p1
                normal = np.array([-segment[1], segment[0]])  # Perpendicular to segment
                denom = np.dot(ray_direction, normal)
                
                if abs(denom) > 1e-10:  # Not parallel
                    t = np.dot(p1 - self.center, normal) / denom
                    if t > 0:  # Intersection is in front of ray
                        intersection = self.center + t * ray_direction
                        # Check if intersection point lies on segment
                        segment_t = np.dot(intersection - p1, segment) / np.dot(segment, segment)
                        if 0 <= segment_t <= 1:
                            dist = t
                            if dist < min_dist:
                                min_dist = dist
                                best_intersection = intersection
            
            if best_intersection is not None:
                projected[i] = best_intersection     # Projected points on boundary
                distances[i] = min_dist
            else:
                # Fallback if no intersection found
                projected[i] = point
                distances[i] = np.linalg.norm(point - self.center)    # Distances from center to projected points
        return projected, distances


class ObstacleFlowField:
    def __init__(self, boundary_points):
        self.boundary_points = boundary_points
        find_center = ObstacleCenterEstimator(boundary_points)
        self.center, _ = find_center.estimate_with_pca()
        
        # Initialize a single GP for both coordinates
        # kernel = ConstantKernel(constant_value=10)  * RBF(4*np.ones(2)) + WhiteKernel(0.01)
        kernel = ConstantKernel(constant_value=25.0) * RBF(length_scale=4.0) + WhiteKernel(noise_level=0.01)
        self.gp = GaussianProcess(kernel=kernel, alpha=0.01)  # This is the regularization parameter)

    def project_using_sdf(self, points, max_iterations=100, tolerance=1e-6):
        sdf_calc = SDFCalculator(self.boundary_points)
        projected = points.copy()
        
        for _ in range(max_iterations):
            # Calculate current distances and gradients
            distances = sdf_calc.signed_distance(projected)
            if np.all(np.abs(distances) < tolerance):
                break
                
            gradients = sdf_calc.calculate_gradient(projected)
            
            # Move points along gradient
            step_size = distances[:, np.newaxis]
            projected -= step_size * gradients
        return projected
  
    """Project points onto boundary using Radial Surface Projector"""
    def radial_projection(self, points):
        projector = RadialSurfaceProjector(self.boundary_points, self.center)
        projected_points, dist_to_boundary = projector.project_points(points)
        return projected_points

    """Project points onto boundary using Radial Surface Projector"""
    def hyper_sphere_projection(self, points):
        projector = HypersphericalProjector(self.boundary_points, self.center)
        projected_points, dist_to_boundary = projector.project_points(points)
        return projected_points
    
    # Inflate the boundary points uniformly outward by a given distance.   
    def inflate_boundary(self, contour_points, inflation_distance):
        """
        contour_points (np.ndarray): Array of boundary points forming the contour
        inflation_distance (float): Distance to inflate the boundary
        """
        if len(contour_points) < 2:
            return np.array([])
            
        # Calculate center of the shape
        center = np.mean(contour_points, axis=0)
        print(f"Shape center: {center}")
        
        inflated_points = []
        n_points = len(contour_points)
        
        for i in range(n_points):
            current_point = contour_points[i]
            
            # Calculate radial direction from center to point
            radial_dir = current_point - center
            # Normalize the radial direction
            radial_dir = radial_dir / (np.linalg.norm(radial_dir) + 1e-10)
            
            # Inflate point along radial direction
            inflated_point = current_point + radial_dir * inflation_distance
            inflated_points.append(inflated_point)
        
        inflated_points = np.array(inflated_points)
        return inflated_points
    
    """Learn flow field mapping from interior points to boundary"""
    def learn_flow_field(self, points_inside):
        # Get boundary projections
        self.projected_boundary_points = self.radial_projection(points_inside)
        # self.projected_boundary_points = self.hyper_sphere_projection(points_inside)
        # self.projected_boundary_points = self.project_using_sdf(points_inside)

        # Inflate boundary points
        inflate_projected_points = self.inflate_boundary(self.projected_boundary_points, 0.25)

        # Calculate displacement vectors
        displacements = self.projected_boundary_points - points_inside

        # calculate the displacement magnitude
        disp_norm = np.linalg.norm(displacements, axis=1)

        unit_displacements = displacements / disp_norm[:, np.newaxis]
        
        # Fit single GP to learn the mapping
        self.gp.fit(points_inside, displacements)

    """Get maximum distance from center to any boundary point."""
    def get_max_distance(self):
        distances = np.linalg.norm(self.boundary_points - self.center, axis=1)
        return np.max(distances)

    def transform_space(self, points):
        points = np.array(points)
        max_dist = self.get_max_distance()  # Maximum distance from center to boundary
        distances = np.linalg.norm(points - self.center, axis=1)
        influence_mask = distances <= max_dist * 2.0  # limit infulece to points within 2.0x max distance
        
        transformed_points = points.copy()
        disp_mag = np.zeros(len(points))
        uncertainties = np.zeros((len(points), 2))  # Initialize with correct shape
        
        if np.any(influence_mask):
            points_to_transform = points[influence_mask]
            
            # Get predictions with covariance
            displacements, cov = self.gp.predict(points_to_transform, return_std=True)  # return_std=True to get covariance
            
            disp_mag[influence_mask] = np.linalg.norm(displacements, axis=1)

            # Extract standard deviation (taking diagonal elements if full covariance returned)
            if cov.ndim > 1:
                std = np.sqrt(np.diag(cov))
            else:
                std = cov
                
            # Calculate smooth scaling factors
            distances_scaled = distances[influence_mask] / max_dist
            scale_factors = np.exp(-0.1 * (distances_scaled) ** 2)   # using Gaussian radial basis function (RBF) for smooth scaling
            
            # Apply scaled transformations
            transformed_points[influence_mask] = points_to_transform + displacements #* scale_factors[:, np.newaxis]
                
            # Scale uncertainties
            uncertainties[influence_mask] = np.outer(scale_factors, np.ones(2)) * std
        return transformed_points, disp_mag, uncertainties
    
    # def transform_space(self, points):
    #     points = np.array(points)
    #     max_dist = self.get_max_distance()
    #     distances = np.linalg.norm(points - self.center, axis=1)
    #     influence_mask = distances <= max_dist * 2.0
        
    #     transformed_points = points.copy()
    #     uncertainties = np.zeros((len(points), 2))
        
    #     if np.any(influence_mask):
    #         points_to_transform = points[influence_mask]
    #         displacements, std = self.gp.predict(points_to_transform, return_std=True)
            
    #         # Compute radial directions (from center to points)
    #         radial_dirs = points_to_transform - self.center
    #         radial_dirs = radial_dirs / (np.linalg.norm(radial_dirs, axis=1)[:, np.newaxis] + 1e-10)
            
    #         # Compute tangential directions (rotate radial by 90 degrees)
    #         tangential_dirs = np.column_stack([-radial_dirs[:, 1], radial_dirs[:, 0]])
            
    #         # Create uncertainty-based scaling
    #         uncertainty_magnitude = np.mean(std, axis=1)
    #         uncertainty_scale = 1.0 + 1.0 * (uncertainty_magnitude / np.max(uncertainty_magnitude))
            
    #         # Distance-based scaling
    #         distances_scaled = distances[influence_mask] / max_dist
    #         radial_scale = np.exp(-0.1 * distances_scaled ** 2)[:, np.newaxis]
            
    #         # Decompose displacement into radial and tangential components
    #         radial_component = np.sum(displacements * radial_dirs, axis=1)[:, np.newaxis] * radial_dirs
    #         tangential_component = np.sum(displacements * tangential_dirs, axis=1)[:, np.newaxis] * tangential_dirs
            
    #         # Use uncertainty scale to increase tangential flow in high uncertainty regions
    #         proximity_factor = np.clip(1.0 - distances_scaled, 0, 1)[:, np.newaxis]
    #         uncertainty_weight = uncertainty_scale[:, np.newaxis]
            
    #         # Combine components with uncertainty weighting
    #         final_displacement = (
    #             radial_component * (1 - proximity_factor) + 
    #             tangential_component * (1 + proximity_factor * uncertainty_weight)
    #         ) * radial_scale
            
    #         transformed_points[influence_mask] = points_to_transform + final_displacement
    #         uncertainties[influence_mask] = uncertainty_magnitude[:, np.newaxis] * radial_scale
        
    #     return transformed_points, uncertainties
    
    # def transform_velocity(self, points, transformed_points, velocities):
    #     J_phi = self.gp.derivative(points)

    #     # # Compute damped dynamics for smoother behavior
    #     # J_phi = J_phi / (1.0 + 0.5 * np.linalg.norm(J_phi, axis=1)[:, np.newaxis])

    #     # Project velocities to be tangent near boundary
    #     distances = np.linalg.norm(transformed_points - self.center, axis=1)
    #     near_boundary = distances <= self.get_max_distance() * 2.0
        
    #     if np.any(near_boundary):
    #         velocities[near_boundary] += np.einsum('ijk,ik->ij', J_phi[near_boundary], velocities[near_boundary])
    #     return velocities      

    def transform_velocity(self, points, disp_mag, transformed_points, velocities):
        # Get Jacobian of the transformation
        J_phi = self.gp.derivative(points)
        
        # Compute distances to center
        distances = np.linalg.norm(transformed_points - self.center, axis=1)
        
        # Define masks for different regions
        near_boundary = distances <= self.get_max_distance() * 1.5
        
        # Transform velocities only in near boundary region
        if np.any(near_boundary):
            # Smooth transition factor
            transition_factor = 1/disp_mag[near_boundary]
            delta_V = np.einsum('ijk,ik->ij', J_phi[near_boundary], velocities[near_boundary]) * transition_factor[:, np.newaxis]
            velocities[near_boundary] = velocities[near_boundary] +  delta_V
        return velocities

"""Estimate center of an obstacle from boundary points."""
class ObstacleCenterEstimator:
    def __init__(self, boundary_points):
        self.boundary_points = boundary_points
    
    """Estimate center using simple mean of points."""
    def estimate_centroid(self):
        return np.mean(self.boundary_points, axis=0)

    """Estimate center using PCA (Useful for elongated shapes)."""
    def estimate_with_pca(self):
        pca = PCA(n_components=2)
        pca.fit(self.boundary_points)
        
        # Center is the mean
        center = pca.mean_
        
        # Get orientation from first principal component
        orientation = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        return center, orientation


"""Project points onto boundary using Radial Surface Projector"""
def radial_projection(points, boundary_points):
    center, _ = ObstacleCenterEstimator(boundary_points).estimate_with_pca()
    projector = RadialSurfaceProjector(boundary_points, center)
    projected_points, dist_to_boundary = projector.project_points(points)
    return projected_points

# Inflate the boundary points uniformly outward by a given distance.   
def inflate_boundary(contour_points, inflation_distance):
    """
    contour_points (np.ndarray): Array of boundary points forming the contour
    inflation_distance (float): Distance to inflate the boundary
    """
    if len(contour_points) < 2:
        return np.array([])
        
    # Calculate center of the shape
    center = np.mean(contour_points, axis=0)
    print(f"Shape center: {center}")
    
    inflated_points = []
    n_points = len(contour_points)
    
    for i in range(n_points):
        current_point = contour_points[i]
        
        # Calculate radial direction from center to point
        radial_dir = current_point - center
        # Normalize the radial direction
        radial_dir = radial_dir / (np.linalg.norm(radial_dir) + 1e-10)
        
        # Inflate point along radial direction
        inflated_point = current_point + radial_dir * inflation_distance
        inflated_points.append(inflated_point)
    
    inflated_points = np.array(inflated_points)
    return inflated_points

def sample_in_polygon_convex(points, num_samples):
    """
    Sample points within a convex polygon using convex combinations
    """
    points = np.array(points)
    n_points = len(points)
    
    # Generate random weights
    weights = np.random.random((num_samples, n_points))
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Generate samples using weighted combinations
    samples = weights @ points
    return samples