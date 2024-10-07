import numpy as np

# MSEを求める
def calculate(target: np.ndarray, target_localized: np.ndarray) -> float:
  return ((target[0] - target_localized[0])**2 + (target[1] - target_localized[1])**2) # 誤差が大きいほどMSEは大きくなる
