import math
import random
import numpy as np

from basis import awgn

def calculate(channel: dict, max_distance_measurement: int, distance_accurate: float):
  tx_power: float = channel["tx_power"]
  path_loss_at_the_reference_distance: float = channel["path_loss_at_the_reference_distance"]
  path_loss_exponent: float = channel["path_loss_exponent"]
  receiver_sensitivity_threthold: float = channel["receiver_sensitivity_threshold"]
  distance_error: float = channel["distance_error"]

  shadowing_standard_deviation: float = eval(channel["shadowing"]["standard_deviation"])

  distances_measured_list = np.array([])
  rx_power_list = np.array([])
  # distance_measured: float = 0.0

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
        # distance_measured += distance_accurate + noise + channel["los"]["mean"]*math.log(1.0 + distance_accurate)
        # distance_measured += distance_accurate # debug
        distance_measured = distance_accurate + noise + channel["los"]["mean"]*math.log(1.0 + distance_accurate)
        # distance_measured = distance_accurate # debug
        if distance_measured < 0.0:
          distance_measured = 0.0
      else:
        rx_power = np.nan
        distance_measured = distance_error
        # break
      distances_measured_list = np.append(distances_measured_list, distance_measured)
      rx_power_list = np.append(rx_power_list, rx_power)
    else:
      distance_measured = distance_error
      rx_power = np.nan
      break
      noise: float = awgn.calculate(channel["nlos"]["standard_deviation"])
      distance_measured += distance_accurate + noise + channel["nlos"]["mean"]*math.log(1.0 + distance_accurate)
  else:
    distance_measured_avg = np.mean(distances_measured_list[np.isfinite(distances_measured_list)]) if distances_measured_list[np.isfinite(distances_measured_list)].size > 0 else -np.inf
    rx_power_avg = np.nanmean(rx_power_list) if rx_power_list[~np.isnan(rx_power_list)].size > 0 else np.nan
  return distance_measured_avg, rx_power_avg
