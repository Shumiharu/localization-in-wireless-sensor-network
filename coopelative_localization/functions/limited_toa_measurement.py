import math

from array import array
import random
import numpy as np

from functions import awgn

def value(target: dict, sensor: dict, sensor_origin: dict, los: dict, max_test: int, max_distance: float, error_distance: float)-> dict:
  avg_measured: array = np.array([0.0]*len(sensor_origin))
  new_avg_measured: array = []
  new_sensor: array = [] 
  for i in range(len(sensor_origin)):
    estimated: float = math.sqrt((target["x"] - sensor_origin[i]["x"])**2.0 + (target["y"] - sensor_origin[i]["y"])**2.0) # 実際の距離
    if(estimated > max_distance):
      avg_measured[i] = error_distance
    else:
      # 以下，試行回数分実測値出して平均を出す
      total_mesured: float = 0.0
      # NLOSならerror_distance 一時的にすべてtrue
      if random.uniform(0,1) > 1.0:
        avg_measured[i] = error_distance
        continue
      # max_testが大きいほど再送回数が多くなる => 1回
      for test in range(max_test):
        # noise: float = awgn.value(los["sigma"]*(math.log(1.0 + estimated)))
        noise: float = awgn.value(los["sigma"])
        total_mesured += estimated + noise + (los["avg"]*math.log(1.0 + estimated))
        # total_mesured += estimated
      if(total_mesured) < 0.0:
        total_mesured = 0.0
      avg_measured[i] = total_mesured/max_test
      
      new_avg_measured.append(total_mesured/max_test)
      new_sensor.append(sensor[i]) # TNの誤差ありを追加
      # new_sensor.append(sensor_origin[i]) # TNの実際の座標を追加
  return {"avg_measured": new_avg_measured, "sensor": new_sensor}
