import math
from array import array

def value(access_point: array, target: dict, transmit_power: float, path_loss_per_meter: float, path_loss_exponent: float) -> dict:
  rss: array = []
  for i in range(len(access_point)):
    distance = math.sqrt((access_point[i]["x"] - target["x"])**2 + (access_point[i]["y"] - target["y"])**2)
    rss.append(transmit_power - (path_loss_per_meter + 10*path_loss_exponent*math.log10(distance)))
  else:
    return rss