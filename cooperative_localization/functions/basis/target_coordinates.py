import numpy as np

from basis import line_of_position
from basis import newton_raphson
from basis import normalization

def calculate(sensors_available: np.ndarray, distances_estimated: np.ndarray, newton_raphson_max_loop: int, newton_raphson_threshold: int, field_range: dict):
  target_estimated = line_of_position.calculate(sensors_available, distances_estimated) # Line of Positionによる初期解の算出
  target_estimated = newton_raphson.calculate(sensors_available, distances_estimated, target_estimated, newton_raphson_max_loop, newton_raphson_threshold) # Newton Raphson法による最適解の算出
  target_estimated = normalization.calculate(field_range, target_estimated) # 測位フィールド外に測位した場合の補正
  return target_estimated