import numpy as np

def calculate(field_range: np.ndarray, target: np.ndarray) -> np.ndarray:
  if(target[0] < field_range["x_bottom"]):
    target[0] = field_range["x_bottom"]
  elif(target[0] > field_range["x_top"]):
    target[0] = field_range["x_top"]
  if(target[1] < field_range["y_bottom"]):
    target[1] = field_range["y_bottom"]
  elif(target[1] > field_range["y_top"]):
    target[1] = field_range["y_top"]
  return target