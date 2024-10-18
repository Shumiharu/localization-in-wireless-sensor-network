import numpy as np

# フィールド中心と測位可能なセンサの重心の距離
def calculate(field_range, sensors_available: np.ndarray):
  center_of_field = np.array([np.mean(np.array([field_range["x_top"], field_range["x_bottom"]])), np.mean(np.array([field_range["y_top"], field_range["y_bottom"]]))])
  centroid = np.array([np.mean(sensors_available[:, 0]), np.mean(sensors_available[:, 1])])
  distance_to_centroid_of_sn_available = np.linalg.norm(centroid - center_of_field)
  return distance_to_centroid_of_sn_available

# Exaple Usage
field_range = {
  "x_top": 30.0,
  "x_bottom": 0.0,
  "y_top": 30.0,
  "y_bottom": 0.0
}
sensors_available = np.array([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0], [10.0, 20.0, 1.0], [20.0, 20.0, 1.0]])
print(f"area_of_triangle = {calculate(field_range, sensors_available)}")
