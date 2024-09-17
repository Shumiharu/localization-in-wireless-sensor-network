from array import array
from functions import rss_measurement

def value(access_point: array, rows: int, columns: int, interval: float, transmit_power: float, path_loss_per_meter: float, path_loss_exponent: float) -> dict:
  # the sample RSS vector -> サンプルのRSS値 ガウス分布に従う
  fingerprint: array = []
  for row in range(1, int(rows)):
    for column in range(1, int(columns)):
      target = {"x": row*interval,"y": column*interval}
      rss = rss_measurement.value(access_point, target, transmit_power, path_loss_per_meter, path_loss_exponent)
      fingerprint.append({"x": row*interval, "y": column*interval, "rss": rss})
  return fingerprint