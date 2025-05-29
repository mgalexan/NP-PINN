import numpy as np

class Tumor:
    """ Generic Class to handle tumor behaviour"""
    def apply_tumor(self, pos: np.ndarray) -> np.ndarray:
        pass


class SphericalTumor(Tumor):
    """ Spherically shaped tumors """
    def __init__(self, center: np.ndarray, r: float):
        super().__init__()
        self.center = center
        self.r = r

    def apply_tumor(self, pos: np.ndarray) -> np.ndarray:
        checked_indices = np.sum((pos - self.center)**2, -1) < self.r
        checked_indices = np.transpose(checked_indices)
        return checked_indices