from array import array
import numpy as np
import math

from functions import coordinate_as_dict
from functions import coordinate_as_list

# Newton Raphson法（座標, 閾値）->（座標）
def value(sensor: dict, coordinate: dict, avg_mesured: array, threshold: float, max_nr: int) -> dict:
  nr_solved_coordinate: dict = coordinate #Newton Raphson法の解

  rad = 0.1
  pol = 0.1

  max_rad_count = 5
  max_pol_count = int(1/pol)

  for rad_count in range(max_rad_count + 1):
    for pol_count in range(max_pol_count):
      count: int =  0
      candidate_coordinate: dict = {"x": coordinate["x"] + rad*rad_count*math.cos(2*math.pi*pol*pol_count), "y": coordinate["y"] + rad*rad_count*math.sin(2*math.pi*pol*pol_count)}
      # Newton Raphsonの処理
      for nr in range(max_nr):
        Q_matrix = np.zeros((2,2)) 

        f: float = 0.0 
        g: float = 0.0
        
        # 各センサについて
        for i in range(len(sensor)):
          diff_x = candidate_coordinate["x"] - sensor[i]["x"]
          diff_y = candidate_coordinate["y"] - sensor[i]["y"]
          d: float = (diff_x)**2 + (diff_y)**2 # 距離差
          f += (float)(2*(1-avg_mesured[i]/math.sqrt(d))*np.squeeze(diff_x)) # 15のx
          g += (float)(2*(1-avg_mesured[i]/math.sqrt(d))*np.squeeze(diff_y)) # 15のy
          Q_matrix[0, 0] += (float)(2*(avg_mesured[i]/(d**1.5))*(np.squeeze(diff_x))**2 + 2*(1-avg_mesured[i]/math.sqrt(d))) # fをxで偏微分
          Q_matrix[0, 1] += (float)(2*(avg_mesured[i]/(d**1.5))*np.squeeze(diff_x)*np.squeeze(diff_y)) # fをyで偏微分
          Q_matrix[1, 0] += (float)(2*(avg_mesured[i]/(d**1.5))*np.squeeze(diff_x)*np.squeeze(diff_y)) # gをxで偏微分
          Q_matrix[1, 1] += (float)(2*(avg_mesured[i]/(d**1.5))*(np.squeeze(diff_y))**2 + 2*(1-avg_mesured[i]/np.sqrt(d))) # gをyで偏微分
        differentiated_function: array = [f, g] # x,yで偏微分された関数
        temporary_candidate_coordinate = coordinate_as_list.value(candidate_coordinate) - np.linalg.inv(Q_matrix)@differentiated_function
        # 距離が閾値より小さいか
        index_x = abs((temporary_candidate_coordinate - coordinate_as_list.value(candidate_coordinate))[0])
        index_y = abs((temporary_candidate_coordinate - coordinate_as_list.value(candidate_coordinate))[1])
        if(index_x < threshold and index_y < threshold):
          nr_solved_coordinate = coordinate_as_dict.value(temporary_candidate_coordinate)
          break
        candidate_coordinate = coordinate_as_dict.value(temporary_candidate_coordinate)
      else:
        continue
      break
    else:
      continue
    break
  return nr_solved_coordinate