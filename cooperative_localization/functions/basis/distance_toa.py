import math
import random
import numpy as np

# ランダムシードの設定
# random.seed(42)
# np.random.seed(42)

from basis import awgn

def calculate(channel: dict, max_distance_measurement: int, distance_accurate: float):
  tx_power: float = channel["tx_power"]
  path_loss_at_the_reference_distance: float = channel["path_loss_at_the_reference_distance"]
  path_loss_exponent: float = channel["path_loss_exponent"]
  receiver_sensitivity_threthold: float = channel["receiver_sensitivity_threshold"]
  distance_error: float = channel["distance_error"]

  shadowing_standard_deviation: float = eval(channel["shadowing"]["standard_deviation"])

  distance_estimated: float = 0.0

  if distance_accurate == 0.0:
    distance_accurate = 10**(-8)
  
  for distance_measurement in range(max_distance_measurement):
    is_los = channel["los"]["probability"] > random.uniform(0, 1)
    if is_los:
      pass_loss: float = 0.0
      shadowing: float = awgn.calculate(shadowing_standard_deviation)
      
      if shadowing >= 0.0:
        pass_loss = path_loss_at_the_reference_distance + 10*path_loss_exponent*math.log10(distance_accurate) + math.log10(shadowing)*10
      else:
        pass_loss = path_loss_at_the_reference_distance + 10*path_loss_exponent*math.log10(distance_accurate) - math.log10(shadowing*(-1))*10
      # pass_loss = path_loss_at_the_reference_distance + 10*path_loss_exponent*math.log10(distance_accurate)

      rx_power = tx_power - pass_loss
      if rx_power > receiver_sensitivity_threthold:
        noise = awgn.calculate(channel["los"]["standard_deviation"])
        distance_estimated += distance_accurate + noise + channel["los"]["mean"]*math.log(1.0 + distance_accurate)
        # distance_estimated += distance_accurate # debug
      else: # 現状ではmax_distance_measurementが1回のため問題ないが，複数回測距を行う場合はこの処理は注意が必要
        distance_estimated = distance_error
        rx_power = np.nan
        break
    else:
      distance_estimated = distance_error
      rx_power = np.nan
      break
      noise: float = awgn.calculate(channel["nlos"]["standard_deviation"])
      distance_estimated += distance_accurate + noise + channel["nlos"]["mean"]*math.log(1.0 + distance_accurate)
  else:
    if distance_estimated < 0.0:
      distance_estimated = 0.0
    distance_estimated /= max_distance_measurement
  return distance_estimated, rx_power

# Example Usage
# import awgn