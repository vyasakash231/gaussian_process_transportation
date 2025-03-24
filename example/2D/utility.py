import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d, LinearNDInterpolator, NearestNDInterpolator
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from policy_transportation import GaussianProcess 


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

"""Project points onto boundary using Radial Surface Projector"""
def radial_projection(points, boundary_points):
    center, _ = ObstacleCenterEstimator(boundary_points).estimate_with_pca()
    projector = RadialSurfaceProjector(boundary_points, center)
    projected_points, dist_to_boundary = projector.project_points(points)
    return projected_points

def sample_boundary_points(boundary_points, n_points):
    """
    Sample points between boundary points to get exactly n total points,
    ensuring original boundary points are included and points are in sequence.
    """
    boundary_points = np.array(boundary_points)
    n_original = len(boundary_points)
    
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

# def generate_inner_contours(boundary_points, n_points, m_contours):
#     """
#     Generate boundary and inner contour points with exactly specified number of points.
#     Points are generated in sequence to maintain proper connectivity.
#     """
#     # Calculate center point
#     center = np.mean(boundary_points, axis=0)
    
#     # Sample boundary points
#     sampled_boundary = sample_boundary_points(boundary_points, n_points)
    
#     # Generate inner contours
#     inner_contours = []
    
#     for i in range(1, m_contours + 1):
#         # Scale factor from center (smaller factor = closer to center)
#         scale = i / (m_contours + 1)
        
#         # Generate points for this contour by scaling from boundary
#         contour_points = center + scale * (sampled_boundary - center)
#         inner_contours.insert(0, contour_points)
#     inner_contours.insert(0, sampled_boundary)
#     return np.vstack(inner_contours)  # Convert list of arrays into a single array

def generate_inner_contours(boundary_points, n_points, m_contours):
    """
    Generate boundary and inner contour points with exactly specified number of points.
    Points are generated in sequence to maintain proper connectivity.
    """
    # Calculate center point
    center = np.mean(boundary_points, axis=0)
    
    # Sample boundary points
    outer_contour_points = sample_boundary_points(boundary_points, n_points)
    
    # Generate inner contours
    contours = outer_contour_points
    for i in range(1, m_contours + 1):
        # Scale factor from center (smaller factor = closer to center)
        scale = i / (m_contours + 1)
        
        # Generate points for this contour by scaling from boundary
        contour_points = center + scale * (outer_contour_points - center)
        contours = np.vstack((contours, contour_points))
    return contours  # Convert list of arrays into a single array

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




