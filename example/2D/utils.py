import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d, LinearNDInterpolator, NearestNDInterpolator
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from policy_transportation import GaussianProcess 


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
        self.find_center = ObstacleCenterEstimator(boundary_points)
        self.center, _ = self.find_center.estimate_with_pca()
        
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
        
    def smooth_displacement_field(self, points, displacement_vectors, sigma=0.25):
        """
        Apply Gaussian smoothing to displacement field.
        """
        # Compute pairwise distances between points
        distances = cdist(points, points)
        
        # Compute Gaussian weights
        weights = np.exp(-distances**2 / (2 * sigma**2))
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        
        # Apply smoothing
        smoothed_displacements = np.dot(weights, displacement_vectors)
        return smoothed_displacements
    
    """Learn flow field mapping from interior points to boundary"""
    def learn_flow_field(self, points_inside):
        # Get boundary projections
        self.projected_boundary_points = self.radial_projection(points_inside)
        # self.projected_boundary_points = self.project_using_sdf(points_inside)

        # Calculate displacement vectors
        displacements = self.projected_boundary_points - points_inside

        # # Apply distance-based normalization
        # normalized_displacements = self.find_center.normalize_displacement_vectors(points_inside, self.projected_boundary_points)
        
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
        
        self.transformed_points = points.copy()
        uncertainties = np.zeros((len(points), 2))  # Initialize with correct shape
        
        if np.any(influence_mask):
            points_to_transform = points[influence_mask]
            
            # Get predictions with covariance
            displacements, cov = self.gp.predict(points_to_transform, return_std=True)  # return_std=True to get covariance
            
            # Extract standard deviation (taking diagonal elements if full covariance returned)
            if cov.ndim > 1:
                std = np.sqrt(np.diag(cov))
            else:
                std = cov
            
            # Apply scaled transformations
            self.transformed_points[influence_mask] = points_to_transform + displacements
                
            # Scale uncertainties
            uncertainties[influence_mask] = std
        return self.transformed_points, uncertainties
    
    # def transform_velocity(self, points, original_velocities):
    #     # Get Jacobian from GP
    #     J_phi = self.gp.derivative(points)
        
    #     # Calculate new velocities using Jacobian (V_new = V + J*V)
    #     delta_vel = np.einsum('ijk,ik->ij', J_phi, original_velocities)
    #     new_velocities = original_velocities + delta_vel
        
    #     # Calculate vectors and distances to obstacle center
    #     vectors_to_center = points - self.center
    #     distances = np.linalg.norm(vectors_to_center, axis=1)
        
    #     # Create Gaussian-like influence field
    #     sigma = 1.0  # Controls the spread of the Gaussian field
    #     influence = np.exp(-1.25 * (distances/sigma)**2)[:, np.newaxis]
        
    #     # Compute tangential components
    #     dir_to_center = vectors_to_center / (distances[:, np.newaxis] + 1e-10)
    #     tangent_dir = np.column_stack([-dir_to_center[:, 1], dir_to_center[:, 0]])
        
    #     # Project velocities
    #     tangent_components = np.sum(new_velocities * tangent_dir, axis=1)[:, np.newaxis]
        
    #     # Create avoidance field
    #     avoid_velocities = tangent_dir * np.sign(tangent_components) * np.linalg.norm(new_velocities, axis=1)[:, np.newaxis]
        
    #     # Smoothly blend based on Gaussian influence
    #     final_velocities = (1 - influence) * new_velocities + influence * avoid_velocities
        
    #     # Additional magnitude scaling near obstacle
    #     magnitude_scale = (1 - np.exp(-1.0 * distances))[:, np.newaxis]
    #     final_velocities *= magnitude_scale
    #     return final_velocities
    
    def transform_velocity(self, points, velocities):
        J_phi = self.gp.derivative(points)

        # Project velocities to be tangent near boundary
        max_dist = self.get_max_distance()  # Maximum distance from center to boundary
        distances = np.linalg.norm(self.transformed_points - self.center, axis=1)
        near_boundary = distances <= self.get_max_distance() * 2.0
        
        if np.any(near_boundary):
            # Calculate smooth scaling factors
            distances_scaled = distances[near_boundary]
            sigma = 0.5 * max_dist  # Controls the spread of the Gaussian field
            scale_factors = np.exp(-1.5 * (distances_scaled/sigma) ** 2.0)   # using Gaussian radial basis function (RBF) for smooth scaling
            delta_vel = np.einsum('ijk,ik->ij', J_phi[near_boundary], velocities[near_boundary])

            velocities[near_boundary] += scale_factors[:, np.newaxis] * delta_vel
        return velocities      


"""Estimate center of an obstacle from boundary points."""
class ObstacleCenterEstimator:
    def __init__(self, boundary_points):
        self.boundary_points = boundary_points
        self.n_dims = boundary_points.shape[1]
        self._pca = None
        self._center = None
        self._components = None
        
    def fit_pca(self):
        """Fit PCA to boundary points and store results."""
        if self._pca is None:
            self._pca = PCA(n_components=self.n_dims)
            self._pca.fit(self.boundary_points)
            self._center = self._pca.mean_
            self._components = self._pca.components_
        return self._pca
        
    def estimate_with_pca(self):
        self.fit_pca()
        # Get orientation from first principal component
        orientation = np.arctan2(self._components[0, 1], self._components[0, 0])
        return self._center, orientation
    
    def estimate_dimensions(self):
        self.fit_pca()
        
        # Project points onto principal axes
        centered_points = self.boundary_points - self._center
        projected_on_axes = self._pca.transform(self.boundary_points)
        
        # Calculate dimensions as range along each principal axis
        # Add margin factor to account for sparse sampling
        margin_factor = 1.1  # Adjust this based on your needs
        self.dimensions = margin_factor * (np.max(projected_on_axes, axis=0) - np.min(projected_on_axes, axis=0))
        
        # Calculate local density of points to adjust dimensions
        def estimate_density_factor(points, axis):
            proj = np.dot(points, axis)
            sorted_proj = np.sort(proj)
            avg_dist = np.mean(np.diff(sorted_proj))
            return min(1.0, avg_dist / (np.std(proj) + 1e-6))
        
        # Adjust dimensions based on point density
        for i in range(len(self.dimensions)):
            density_factor = estimate_density_factor(centered_points, self._components[i])
            self.dimensions[i] *= density_factor
            
        return self.dimensions, self._components
    
    def normalize_displacement_vectors(self, interior_points, projected_points):
        self.estimate_dimensions()
        
        # Calculate displacement vectors
        displacement_vectors = projected_points - interior_points
        
        # Transform displacement vectors to principal component space
        displacement_vectors_pca = np.dot(displacement_vectors, self._components.T)
        
        # Avoid division by zero
        min_dimension = 1e-6
        safe_dimensions = np.maximum(self.dimensions, min_dimension)
        
        # Normalize each component by corresponding dimension
        normalized_displacements_pca = displacement_vectors_pca / safe_dimensions
        
        # Transform back to original space
        normalized_displacements = np.dot(normalized_displacements_pca, self._components)
        return normalized_displacements


"""Project points onto boundary using Radial Surface Projector"""
def radial_projection(points, boundary_points):
    center, _ = ObstacleCenterEstimator(boundary_points).estimate_with_pca()
    projector = RadialSurfaceProjector(boundary_points, center)
    projected_points, dist_to_boundary = projector.project_points(points)
    return projected_points

"""Project points onto boundary using Radial Surface Projector"""
def sdf_projection(points, boundary_points, max_iterations=250, tolerance=1e-8):
    center, _ = ObstacleCenterEstimator(boundary_points).estimate_with_pca()
    projector = SDFCalculator(boundary_points)
    projected = points.copy()
    
    for _ in range(max_iterations):
        # Calculate current distances and gradients
        distances = projector.signed_distance(projected)
        if np.all(np.abs(distances) < tolerance):
            break
            
        gradients = projector.calculate_gradient(projected)
        
        # Move points along gradient
        step_size = distances[:, np.newaxis]
        projected -= step_size * gradients
    return projected

def sample_boundary_points(boundary_points, n_points):
    """
    Sample points between boundary points to get exactly n total points,
    ensuring original boundary points are included and points are in sequence.
    """
    boundary_points = np.array(boundary_points)
    n_original = len(boundary_points)
    
    if n_points < n_original:
        raise ValueError(f"n_points ({n_points}) must be >= number of boundary points ({n_original})")
    
    # Calculate points to add between each pair of original points
    points_to_add = n_points - n_original
    segment_lengths = []
    total_length = 0
    
    # Calculate segment lengths
    for i in range(n_original):
        start = boundary_points[i]
        end = boundary_points[(i + 1) % n_original]
        length = np.linalg.norm(end - start)
        segment_lengths.append(length)
        total_length += length
    
    # Distribute points proportionally
    points_per_segment = []
    remaining_points = points_to_add
    
    for i in range(n_original - 1):
        n_points_seg = int((segment_lengths[i] / total_length) * points_to_add)
        points_per_segment.append(n_points_seg)
        remaining_points -= n_points_seg
    
    # Add remaining points to last segment
    points_per_segment.append(remaining_points)
    
    # Generate points
    sampled_points = []
    
    # Process each segment
    for i in range(n_original):
        # Add current boundary point
        sampled_points.append(boundary_points[i])
        
        # Add interpolated points if any
        if points_per_segment[i] > 0:
            start = boundary_points[i]
            end = boundary_points[(i + 1) % n_original]
            
            # Generate intermediate points
            for t in np.linspace(0, 1, points_per_segment[i] + 2)[1:-1]:
                point = start + t * (end - start)
                sampled_points.append(point)
    return np.array(sampled_points)

def generate_inner_contours(boundary_points, n_points, m_contours):
    """
    Generate boundary and inner contour points with exactly specified number of points.
    Points are generated in sequence to maintain proper connectivity.
    """
    # Calculate center point
    center = np.mean(boundary_points, axis=0)
    
    # Sample boundary points
    sampled_boundary = sample_boundary_points(boundary_points, n_points)
    
    # Generate inner contours
    inner_contours = []
    
    for i in range(1, m_contours + 1):
        # Scale factor from center (smaller factor = closer to center)
        scale = i / (m_contours + 1)
        
        # Generate points for this contour by scaling from boundary
        contour_points = center + scale * (sampled_boundary - center)
        inner_contours.insert(0, contour_points)
    inner_contours.insert(0, sampled_boundary)
    return np.vstack(inner_contours)  # Convert list of arrays into a single array

"""Sample points within a polygon defined by N points"""
def sample_in_polygon(points, num_samples):  # points are evenly distributed
    points = np.array(points)
    
    # Triangulate the polygon
    tri = Delaunay(points)
    
    # Calculate areas of all triangles
    triangles = points[tri.simplices]
    areas = np.abs(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])) / 2
    total_area = np.sum(areas)
    
    # Distribute points proportionally to triangle areas
    samples_per_triangle = np.random.multinomial(num_samples, areas/total_area)
    
    samples = []
    for triangle, n_samples in zip(triangles, samples_per_triangle):
        if n_samples > 0:
            # Generate random barycentric coordinates
            r1 = np.random.random(n_samples)
            r2 = np.random.random(n_samples)
            
            # Convert to barycentric coordinates
            sqrt_r1 = np.sqrt(r1)
            barycentric_coords = np.column_stack([1 - sqrt_r1, sqrt_r1 * (1 - r2), sqrt_r1 * r2])
            
            # Convert to Cartesian coordinates
            triangle_samples = barycentric_coords @ triangle
            samples.append(triangle_samples)
    return np.vstack(samples)

"""Sample points within a convex polygon using convex combinations of vertices"""
def sample_in_polygon_convex(points, num_samples):  # points are concentrated at the center
    points = np.array(points)
    n_points = len(points)
    
    # Generate random weights
    weights = np.random.random((num_samples, n_points))
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Generate samples using weighted combinations
    samples = weights @ points
    return samples

def generate_divergent_rotational_flow(boundary_points, points_inside):
    """Generate divergent flow with rotation from obstacle center"""
    center, _ = ObstacleCenterEstimator(boundary_points).estimate_with_pca()

    # Get vectors from center to points
    vectors_from_center = points_inside - center
    
    # Calculate distances from center
    distances = np.linalg.norm(vectors_from_center, axis=1)
    
    # Normalize vectors to get radial directions
    radial_directions = vectors_from_center / (distances[:, np.newaxis] + 1e-10)
    
    # Create rotational component (counterclockwise)
    # For each point, create perpendicular vector for rotation
    rotational_directions = np.column_stack([-radial_directions[:, 1], radial_directions[:, 0]])
    
    # Scale velocities - stronger near center, weaker at boundary
    velocity_scale = np.exp(-0.1 * distances)[:, np.newaxis]
    
    # Combine radial and rotational components
    # You can adjust the ratio between radial and rotational (e.g., 0.7 and 0.3)
    velocities = (0.5 * radial_directions + 0.5 * rotational_directions) * velocity_scale
    return velocities 

def generate_shaped_divergent_flow(boundary_points, points_inside):
    """Generate divergent flow shaped according to obstacle geometry"""
    # Get obstacle shape information using PCA
    obstacle_estimate = ObstacleCenterEstimator(boundary_points)
    center, _ = obstacle_estimate.estimate_with_pca()
    dimensions, components = obstacle_estimate.estimate_dimensions()
    
    # Get vectors from center to points
    vectors_from_center = points_inside - center
    
    # Project vectors onto principal components
    projected_vectors = np.zeros_like(vectors_from_center)
    for i in range(2):  # For 2D
        projected_vectors += np.outer(np.dot(vectors_from_center, components[i]) / dimensions[i], components[i])
    
    # Calculate scaled distances based on obstacle shape
    # Points further along the shorter axis will have larger velocities
    scaled_distances = np.zeros(len(points_inside))
    for i in range(2):
        component_distances = np.abs(np.dot(vectors_from_center, components[i]))
        scaled_distances += (component_distances / dimensions[i]) ** 2
    scaled_distances = np.sqrt(scaled_distances)
    
    # Normalize vectors to get radial directions
    radial_directions = projected_vectors / (np.linalg.norm(projected_vectors, axis=1)[:, np.newaxis] + 1e-10)
    
    # Create rotational component (counterclockwise)
    rotational_directions = np.column_stack([-radial_directions[:, 1], radial_directions[:, 0]])
    
    # Scale velocities - stronger near center, weaker at boundary
    # Use scaled distances for more appropriate falloff
    velocity_scale = np.exp(-0.5 * scaled_distances)[:, np.newaxis]
    
    # Calculate shape-based weighting for radial vs rotational
    # More elongated shapes get more rotation
    shape_ratio = min(dimensions) / max(dimensions)
    radial_weight = 0.2 + 0.3 * shape_ratio  # Will be between 0.5 and 0.8
    rotational_weight = 1 - radial_weight
    
    # Combine radial and rotational components with shape-based weights
    velocities = (radial_weight * radial_directions + rotational_weight * rotational_directions) * velocity_scale
    return velocities

