import numpy as np

# 測位可能なセンサの重心とターゲットの推定座標との距離
def calculate(sensors_available: np.ndarray, target_estimated: np.ndarray):
  centroid = np.array([np.mean(sensors_available[:, 0]), np.mean(sensors_available[:, 1])])
  distance_from_centroid_of_sn_available_to_tn_estimated = np.linalg.norm(centroid - target_estimated[:2])
  return distance_from_centroid_of_sn_available_to_tn_estimated