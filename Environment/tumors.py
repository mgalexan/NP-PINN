import numpy as np
from Environment.geometry import GeometrySpace

class Tumor:
    """ Generic Class to handle tumor behaviour"""
    def apply_tumor(self, geo: GeometrySpace) -> np.ndarray:
        pass


class SphericalTumor(Tumor):
    """ Spherically shaped tumors """
    def __init__(self, center: np.ndarray, r: float):
        super().__init__()
        self.center = center
        self.r = r

    def apply_tumor(self, geo: GeometrySpace) -> np.ndarray:
        if not(isinstance(geo.coord_matrix, np.ndarray)):
            print("Error: Initialize geometry coordinates with get_coordinate_matrix before calculating tumor locations")
            return None
        
        pos = geo.coord_matrix
        checked_indices = np.sum((pos - self.center)**2, -1) < self.r**2
        checked_indices = np.transpose(checked_indices)
        return checked_indices