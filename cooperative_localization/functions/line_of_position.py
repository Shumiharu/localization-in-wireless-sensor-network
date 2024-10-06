from array import array
import numpy as np
from functions import coordinate_as_dict

def value(sensor: dict, avg_measured: array) -> dict:
  # R 
  R_matrix = np.zeros((len(sensor) - 1 , 2))
  for i in range(len(sensor) - 1):
    R_matrix[i, 0] = 2*(sensor[(len(sensor) - 1)]["x"] - sensor[i]["x"])
    R_matrix[i, 1] = 2*(sensor[(len(sensor) - 1)]["y"] - sensor[i]["y"])
  
  # I
  I_matrix = np.zeros((len(sensor) - 1 , 1))
  for i in range(len(sensor) - 1):
    I_matrix[i] = -(sensor[i]["x"])**2.0 - (sensor[i]["y"])**2.0 + (avg_measured[i])**2.0\
       - (-(sensor[(len(sensor) - 1)]["x"])**2.0 - (sensor[(len(sensor) - 1)]["y"])**2 + (avg_measured[(len(sensor) - 1)])**2.0)

  # (R^T*R)^(-1)*I
  lop_solved_coordinate = coordinate_as_dict.value(np.linalg.inv(R_matrix.T@R_matrix)@(R_matrix.T)@I_matrix)

  return lop_solved_coordinate