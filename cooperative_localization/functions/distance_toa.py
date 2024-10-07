import math
import random
import numpy as np

from array import array
from functions import awgn

def calculate(channel: dict, max_distance_measurement: int, distance_accurate: float)-> dict:
  tx_power: float = channel["tx_power"]
  path_loss_at_the_reference_distance: float = channel["path_loss_at_the_reference_distance"]
  path_loss_exponent: float = channel["path_loss_exponent"]
  receiver_sensitivity_threthold: float = channel["receiver_sensitivity_threshold"]
  distance_error: float = channel["distance_error"]

  shadowing_standard_deviation: float = eval(channel["shadowing"]["standard_deviation"])

  distance_estimated: float = 0.0

  if distance_accurate == 0.0:
    distance_accurate = 10**(-8)
  
  for toa_distance in range(max_distance_measurement):
    is_los = channel["los"]["probability"] > random.uniform(0, 1)
    if is_los:
      pass_loss: float = 0.0
      shadowing: float = awgn.calculate(shadowing_standard_deviation)
      
      if shadowing >= 0.0:
        pass_loss = path_loss_at_the_reference_distance + 10*path_loss_exponent*math.log10(distance_accurate) + math.log10(shadowing)*10
      else:
        pass_loss = path_loss_at_the_reference_distance + 10*path_loss_exponent*math.log10(distance_accurate) - math.log10(shadowing*(-1))*10
      rx_power = tx_power - pass_loss
      if rx_power > receiver_sensitivity_threthold:
        noise = awgn.calculate(channel["los"]["standard_deviation"])
        distance_estimated += distance_accurate + noise + channel["los"]["mean"]*math.log(1.0 + distance_accurate)
        # distance_estimated += distance_accurate # debug
      else:
        distance_estimated = distance_error
        break
    else:
      distance_estimated = distance_error
      break
      noise: float = awgn.calculate(channel["nlos"]["standard_deviation"])
      distance_estimated += distance_accurate + noise + channel["nlos"]["mean"]*math.log(1.0 + distance_accurate)
  else:
    if distance_estimated < 0.0:
      distance_estimated = 0.0
    distance_estimated /= max_distance_measurement
  return distance_estimated

  # avg_measured: array = np.array([0.0]*len(sensor_origin))
  # new_avg_measured: array = []
  # new_sensor: array = [] 
  # for i in range(len(sensor_origin)):
  #   estimated: float = math.sqrt((target["x"] - sensor_origin[i]["x"])**2.0 + (target["y"] - sensor_origin[i]["y"])**2.0) # 実際の距離
  #   if(estimated > max_distance):
  #     avg_measured[i] = distance_error
  #   else:
  #     # 以下，試行回数分実測値出して平均を出す
  #     total_mesured: float = 0.0
  #     # NLOSならdistance_error 一時的にすべてtrue
  #     if random.uniform(0,1) > 1.0:
  #       avg_measured[i] = distance_error
  #       continue
  #     # max_testが大きいほど再送回数が多くなる => 1回
  #     for test in range(max_test):
  #       # noise: float = awgn.calculate(los["standard_deviation"]*(math.log(1.0 + estimated)))
  #       noise: float = awgn.calculate(los["standard_deviation"])
  #       total_mesured += estimated + noise + (los["avg"]*math.log(1.0 + estimated))
  #       # total_mesured += estimated
  #     if(total_mesured) < 0.0:
  #       total_mesured = 0.0
  #     avg_measured[i] = total_mesured/max_test
      
  #     new_avg_measured.append(total_mesured/max_test)
  #     new_sensor.append(sensor[i]) # TNの誤差ありを追加
  #     # new_sensor.append(sensor_origin[i]) # TNの実際の座標を追加
  # return {"avg_measured": new_avg_measured, "sensor": new_sensor}
