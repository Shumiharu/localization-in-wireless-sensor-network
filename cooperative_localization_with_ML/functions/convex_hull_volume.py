import numpy as np
from scipy.spatial import ConvexHull

def value(sensors_available: np.ndarray):
    hull = ConvexHull(sensors_available[:, :2])
    return hull.volume