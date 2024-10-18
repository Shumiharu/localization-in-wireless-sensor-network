import numpy as np

# 残差の平均を算出
def calculate(sensors_available: np.ndarray, distances_estimated: np.ndarray, target_estimated: np.ndarray) -> float:
    residual_avg = np.sum((distances_estimated - np.linalg.norm(sensors_available[:, :2] - target_estimated[:2], axis=1))**2)/len(sensors_available)
    return residual_avg

# Example Usage
# sensors_available = np.array([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0], [10.0, 20.0, 1.0], [20.0, 20.0, 1.0]])
# distances_estimated = np.array([ 1.91035687, 10.87991461, 10.47944295, 15.46761316])
# target_estimated = np.array([8.70838408, 9.42094928, 0.0])
# print(f"residual_avg = {calculate(sensors_available, distances_estimated, target_estimated)}")
