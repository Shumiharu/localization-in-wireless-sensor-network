import numpy as np

# センサーの重心から消失ノード（vanishing node: VN）の最大距離を計算する
def calculate(sensors: np.ndarray, distances_measured: np.ndarray) -> float:
    distance_from_centroid_of_sensors_to_vn_maximized = 0.0
    sensors_unavailable = sensors[np.isinf(distances_measured)]
    if not len(sensors_unavailable) == 0:
        centroid = np.array([np.mean(sensors[:, 0]), np.mean(sensors[:, 1])])
        distance_from_centroid_of_sensors_to_vn_maximized = np.max(np.linalg.norm(centroid - sensors_unavailable[:, :2], axis=1))
    return distance_from_centroid_of_sensors_to_vn_maximized

# Example Usage
# sensors = np.array([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0], [10.0, 20.0, 1.0], [20.0, 20.0, 1.0]])
# target_estimated = np.array([20.0180388, 11.52676422, 0.])
# distances_measured = np.array([10.49793249, 0.63111357, np.inf, 8.23798477])
# print(f"distance_from_centroid_of_sensors_to_vn_maximized = {calculate(sensors,distances_measured, target_estimated)}")
