import math

from array import array
import numpy as np

from functions import awgn

def value(target: dict, sensor: dict, los: dict, nlos: dict, max_test: int)-> array:
  avg_mesured: list[float] = np.array([0.0]*len(sensor))
  for i in range(len(sensor)):
    estimated: float = math.sqrt((target["x"] - sensor[i]["x"])**2.0 + (target["y"] - sensor[i]["y"])**2.0) # 実際の距離
    # 以下，試行回数分実測値出して平均を出す
    total_mesured: float = 0.0
    for test in range(max_test):
      noise: float = awgn.value(los["sigma"]*(math.log(1.0 + estimated)) + nlos["sigma"]*sensor[i]["is_nlos"]) # los["sigma"]はm（メートル），math.log(1.0 + estimated)は無次元量なのでメートルが返る
      total_mesured += estimated + noise + (los["avg"]*math.log(1.0 + estimated) + nlos["avg"]*sensor[i]["is_nlos"]) # 同様にlos["sigma"]もm（メートル）なのでこちらもメートルが返る．すなわち平均でlos["avg"] m分だけずれてlos["sigma"] mで細かく変動することになる．
      # total_mesured += estimated + (noise + los["avg"] + nlos["avg"]*sensor[i]["is_nlos"])*math.log(1.0 + estimated)
      # total_mesured += estimated
    if(total_mesured) < 0.0:
      total_mesured = 0.0
    avg_mesured[i] = total_mesured/max_test
  return avg_mesured
