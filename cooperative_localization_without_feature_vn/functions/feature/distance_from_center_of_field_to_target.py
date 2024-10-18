import numpy as np

# フィールドの中心からターゲットまでの距離
def calculate(field_range: dict, target_estimated: np.ndarray):
  center_of_field = np.array([np.mean(np.array([field_range["x_top"], field_range["x_bottom"]])), np.mean(np.array([field_range["y_top"], field_range["y_bottom"]]))])
  return np.linalg.norm(target_estimated[:2] - center_of_field)

# Example Usage
# field_range = {
#   "x_top": 30.0,
#   "x_bottom": 0.0,
#   "y_top": 30.0,
#   "y_bottom": 0.0
# }
# target_estimated = np.array([8.70838408, 9.42094928, 0.0])
# print(f"distance_from_center_of_field_to_target = {calculate(field_range, target_estimated)}")

