import numpy as np

def calculate(field_range: dict, target_localized: np.ndarray):
  return np.sqrt((target_localized[0] - field_range["x_top"])**2 + (target_localized[1] - field_range["y_top"])**2)