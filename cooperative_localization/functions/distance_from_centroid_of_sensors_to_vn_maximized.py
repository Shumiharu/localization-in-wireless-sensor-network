import numpy as np

# センサーの重心から消失ノード（vanishing node: VN）の最大距離を計算する
def calculate(sensors_available: np.ndarray, distances_measured: np.ndarray, target_estimated: np.ndarray) -> float:
    distance_from_centroid_of_sensors_to_vn_maximized = 0.0
    for distance_measured in distances_measured:
        if np.isinf(distance_measured):
            centroid = np.array([np.mean(sensors_available[:, 0]), np.mean(sensors_available[:, 1])])
            distance_from_centroid_of_sensors_to_vn = np.linalg.norm(centroid - target_estimated[:2])
            if distance_from_centroid_of_sensors_to_vn > distance_from_centroid_of_sensors_to_vn_maximized:
                distance_from_centroid_of_sensors_to_vn_maximized = distance_from_centroid_of_sensors_to_vn
    return distance_from_centroid_of_sensors_to_vn_maximized

# Example Usage
# sensors_available = np.array([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0], [10.0, 20.0, 1.0], [20.0, 20.0, 1.0]])
# target_estimated = np.array([20.0180388, 11.52676422, 0.])
# distances_measured = np.array([10.49793249, 0.63111357, np.inf, 8.23798477, 11.86804664])
# print(f"distance_from_centroid_of_sensors_to_vn_maximized = {calculate(sensors_available,distances_measured, target_estimated)}")
