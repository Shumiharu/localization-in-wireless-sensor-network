import numpy as np

def calculate(sensors_available: np.ndarray):
  return np.mean(sensors_available[:, 3] + 1)