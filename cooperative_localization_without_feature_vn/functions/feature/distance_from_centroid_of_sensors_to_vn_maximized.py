import numpy as np
import yaml

# センサーの重心から消失ノード（vanishing node: VN）の最大距離を計算する
# def calculate(sensors: np.ndarray, distances_measured: np.ndarray) -> float:
#     distance_from_centroid_of_sensors_available_to_unavailable_maximized = 0.0
#     sensors_unavailable = sensors[np.isinf(distances_measured)]
#     if not len(sensors_unavailable) == 0:
#         centroid = np.array([np.mean(sensors[:, 0]), np.mean(sensors[:, 1])])
#         distance_from_centroid_of_sensors_available_to_unavailable_maximized = np.max(np.linalg.norm(centroid - sensors_unavailable[:, :2], axis=1))
#     return distance_from_centroid_of_sensors_available_to_unavailable_maximized

def calculate(sensors: np.ndarray, distances_measured: np.ndarray, distance_measured_max: float) -> float:
    isinf_mask = np.isinf(distances_measured)
    sensors_available, sensors_unavailable = sensors[~isinf_mask], sensors[isinf_mask]

    distance_measured_max = np.max(distances_measured)
    distance_from_centroid_of_sensors_available_to_unavailable_maximized = 0.0
    if len(sensors_unavailable) > 0:
        centroid = np.array([np.mean(sensors_available[:, 0]), np.mean(sensors_available[:, 1])]) # 測位可能なセンサーから重心を求める
        distances_from_centroid_of_sensors_available_to_unavailable = np.linalg.norm(centroid - sensors_unavailable[:, :2], axis=1)
        distances_from_centroid_of_sensors_available_to_unavailable = distances_from_centroid_of_sensors_available_to_unavailable[distances_from_centroid_of_sensors_available_to_unavailable < distance_measured_max]
        if len(distances_from_centroid_of_sensors_available_to_unavailable) > 0:
            distance_from_centroid_of_sensors_available_to_unavailable_maximized = np.max(distances_from_centroid_of_sensors_available_to_unavailable)
    return distance_from_centroid_of_sensors_available_to_unavailable_maximized

# def calculate2(sensors: np.ndarray, target_estimated: np.ndarray, distances_measured: np.ndarray, error_threshold: float) -> float:
#     distance_from_centroid_of_sensors_available_to_unavailable_maximized = 0.0
#     sensors_unavailable = sensors[np.isinf(distances_measured)] # 測位できなかったセンサーを抽出
#     centroid = np.array([np.mean(sensors[:, 0]), np.mean(sensors[:, 1])])

#     for sensor_unavailable in sensors_unavailable:
#         distance_unavailable = np.linalg.norm(sensor_unavailable[:2] - target_estimated[:2]) # 測位誤差が大きい場合，測距できる場合があるという特徴量
#         if not np.isinf(distance_unavailable_estimated):
#             distance_from_centroid_of_sensors_to_vn = np.linalg.norm(centroid - sensor_unavailable[:2])
#             if distance_from_centroid_of_sensors_to_vn > distance_from_centroid_of_sensors_available_to_unavailable_maximized:
#                 distance_from_centroid_of_sensors_available_to_unavailable_maximized = distance_from_centroid_of_sensors_to_vn
#     return distance_from_centroid_of_sensors_available_to_unavailable_maximized

# Example Usage
# sensors = np.array([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0], [10.0, 20.0, 1.0], [20.0, 20.0, 1.0]])
# distances_measured = np.array([10.49793249, 0.63111357, -np.inf, 8.23798477])
# print(f"distance_from_centroid_of_sensors_available_to_unavailable_maximized = {calculate(sensors, distances_measured)}")
