import numpy as np
from scipy.spatial import ConvexHull

def calculate(sensors: np.ndarray):
    hull = ConvexHull(sensors[:, :2], qhull_options='QJ')
    return hull.volume

# Example Usage
# sensors = np.array([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0], [10.0, 20.0, 1.0], [20.0, 20.0, 1.0]])
# print(f"result = {calculate(sensors)}")