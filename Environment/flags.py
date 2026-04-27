import numpy as np
from Environment.geometry import GeometrySpace

class Flag:
    """ Generic Class to handle tumor behaviour"""
    def apply_flag(self, geo: GeometrySpace) -> np.ndarray:
        pass


class SphericalFlag(Flag):
    """ Spherically shaped tumors """
    def __init__(self, center, r: float, smoothing_width: float = 0.0):
        super().__init__()
        self.center = center[::-1]
        self.r = np.array(r)
        self.smoothing_width = smoothing_width

    def apply_flag(self, geo: GeometrySpace) -> np.ndarray:
        if not(isinstance(geo.coord_matrix, np.ndarray)):
            print("Error: Initialize geometry coordinates with get_coordinate_matrix before calculating tumor locations")
            return None
        
        pos = geo.coord_matrix
        distances_squared = np.sum((pos - self.center)**2, -1)
        
        if self.smoothing_width > 0:
            # Create smooth transition zone
            inner_r = self.r - self.smoothing_width/2
            outer_r = self.r + self.smoothing_width/2
            
            # Calculate distance from center
            dist_from_center = np.sqrt(np.sum((pos - self.center)**2, -1))
            
            # Initialize result array
            checked_indices = np.zeros_like(dist_from_center)
            
            # Inside inner radius: 1
            checked_indices[dist_from_center <= inner_r] = 1.0
            
            # In transition zone: linear interpolation
            transition_mask = (dist_from_center > inner_r) & (dist_from_center < outer_r)
            if np.any(transition_mask):
                normalized_dist = (dist_from_center[transition_mask] - inner_r) / self.smoothing_width
                checked_indices[transition_mask] = 1 - normalized_dist
            
            # Outside outer radius: 0
            checked_indices[dist_from_center >= outer_r] = 0.0
        else:
            # Sharp cutoff
            checked_indices = (distances_squared <= self.r**2).astype(float)
        
        checked_indices = np.transpose(checked_indices)
        return checked_indices
    
class EllipticalFlag2D(Flag):
    """ Elliptically shaped tumors """
    def __init__(self, center, rx: float, ry: float):
        super().__init__()
        self.center = center[::-1]
        self.rx = np.array(rx)
        self.ry = np.array(ry)


    def apply_flag(self, geo: GeometrySpace) -> np.ndarray:
        if not(isinstance(geo.coord_matrix, np.ndarray)):
            print("Error: Initialize geometry coordinates with get_coordinate_matrix before calculating tumor locations")
            return None
        
        pos = geo.coord_matrix
        checked_indices = ((pos[:,:,0] - self.center[0]) / self.ry)**2 + ((pos[:,:,1] - self.center[1]) / self.rx)**2 < 1
        checked_indices = np.transpose(checked_indices)
        return checked_indices
    
class BoxFlag2D(Flag):
    """ Elliptically shaped tumors """
    def __init__(self, center, x: float, y: float):
        super().__init__()
        self.center = center[::-1]
        self.x = np.array(x)
        self.y = np.array(y)


    def apply_flag(self, geo: GeometrySpace) -> np.ndarray:
        if not(isinstance(geo.coord_matrix, np.ndarray)):
            print("Error: Initialize geometry coordinates with get_coordinate_matrix before calculating tumor locations")
            return None
        
        pos = geo.coord_matrix
        checked_indices = np.logical_and(np.abs(pos[:,:,0] - self.center[0]) < self.y / 2, np.abs(pos[:,:,1] - self.center[1]) < self.x / 2)  
        checked_indices = np.transpose(checked_indices)
        return checked_indices
    
class EdgeFlag2D(Flag):
    def __init__(self, width):
        super().__init__()
        self.width = width
    
    def apply_flag(self, geo: GeometrySpace) -> np.ndarray:
        pos = geo.coord_matrix

        x_high = geo.width - self.width
        x_low = self.width
        y_high = geo.height - self.width
        y_low = self.width

        
        checked_indices =  \
            (pos[:,:,0] >  x_high) | \
            (pos[:,:,0] < x_low) | \
            (pos[:,:,1] >  y_high) | \
            (pos[:,:,1] < y_low)
        
        return checked_indices

class SphericalTaperingFlag(Flag):
    """ Spherically shaped tumors """
    def __init__(self, center, r: float):
        super().__init__()
        self.center = center[::-1]
        self.r = np.array(r)

    def apply_flag(self, geo: GeometrySpace) -> np.ndarray:
        if not(isinstance(geo.coord_matrix, np.ndarray)):
            print("Error: Initialize geometry coordinates with get_coordinate_matrix before calculating tumor locations")
            return None
        
        pos = geo.coord_matrix
        checked_indices = np.sum((pos - self.center)**2, -1) / self.r**2
        checked_indices = np.transpose(checked_indices)
        checked_indices = np.maximum(checked_indices, 1)
        return checked_indices

