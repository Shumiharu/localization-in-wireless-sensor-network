from array import array
from typing import Dict
import numpy as np

def value(dict: dict) -> array:
  coordinate: array = np.array([np.squeeze(dict["x"]), np.squeeze(dict["y"])])
  return coordinate