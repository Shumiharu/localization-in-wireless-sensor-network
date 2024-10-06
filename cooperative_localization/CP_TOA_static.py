# 1点だけTNが協調する測位のプログラム

#       10m
# ○-------------○
# |  ○ ->       |
# |             |
# |      ○(5,5) | 10m
# |             |
# |             |
# ○-------------○
# 
# LOS環境
# TN数: 2
# TN_1 (5,5)
# TN_2 (x,y)
# TN_2（以降）はランダム生成
# TN_1を測位したあとANにしてTN_2を測位（協調測位）
# TN2のRMSEを算出する

from array import array
# from __future__ import annotations
import math 
import random

from functions import toa_mesurement
from functions import field
from functions import line_of_position
from functions import newton_raphson
from functions import mean_squared_error

if __name__ == "__main__":
  print("Cooperative Positioning Time of Arrival")
  all_targets: int = 2
  print("TN: " + str(all_targets) + " static") # TNの数は固定

  is_cooperate: int = 1
  print("Cooperative Mode" if is_cooperate == 1 else "Incooperative Mode")

  max_test: int = 30
  max_loop: int = 10
  max_nr: int = 10

  # 座標定義
  x_top: float = 10.0
  y_top: float = 10.0
  x_bottom: float = 0.0
  y_bottom: float = 0.0

  # 測定範囲
  mesured_range : dict = {
    "x_top": x_top,
    "x_bottom": x_bottom,
    "y_top": y_top,
    "y_bottom": y_bottom
  }

  # LOSとNLOS環境におけるそれぞれの分散と平均とノード
  los: dict = {
    "sigma": 0.269, # 正規化距離誤差
    "avg": 0.21, # 
  }
  nlos: dict = {
    "sigma": 0.809, # 正規化距離誤差
    "avg": 1.62, 
  }

  # 合計RMSE
  total_rmse: float = 0.0

  # 進捗
  progress: int = 0

  # 施行回数
  cycle: int = 10000

  # 固定ターゲット
  static_target: dict = {"x": 5.0, "y": 5.0, "is_nlos": 0}
  
  for h in range(cycle):
    mse: float = 0.0

    # ターゲット
    targets: array = []
    if(is_cooperate):
      for i in range(all_targets):
        if(i == 0):
          targets.append(static_target)
        else:
          targets.append({"x": (float(random.randrange(101)/10)), "y": (float(random.randrange(101)/10)), "is_nlos": 0})
    else:
      targets.append({"x": (float(random.randrange(101)/10)), "y": (float(random.randrange(101)/10)), "is_nlos": 0})

    for loop in range(max_loop):
      # センサ各位置（実際の位置）
      sensor_origin: array = [
        {"x": x_bottom, "y": y_bottom, "is_nlos": 0},
        {"x": x_top, "y": y_bottom, "is_nlos": 0},
        {"x": x_bottom, "y": y_top, "is_nlos": 0},
        {"x": x_top, "y": y_top, "is_nlos": 0},
      ]

       # センサ各位置（デフォルト以外は候補点の座標）
      sensor = [
        {"x": x_bottom, "y": y_bottom, "is_nlos": 0},
        {"x": x_top, "y": y_bottom, "is_nlos": 0},
        {"x": x_bottom, "y": y_top, "is_nlos": 0},
        {"x": x_top, "y": y_top, "is_nlos": 0},
      ]

      # Incooperate Modeの場合，事前にANを追加しておく
      if(is_cooperate == 0):
        sensor_origin.append(static_target)
        sensor.append(static_target)

      for i in range(len(targets)):
        avg_mesured = toa_mesurement.value(targets[i], sensor_origin, los, nlos, max_test) # センサーからの距離を測定
        initial_coordinate: dict = line_of_position.value(sensor, avg_mesured) # Line Of Position で初期値決定
        # initial_coordinate: dict = line_of_position.value(sensor_origin, avg_mesured) 
        converged_coordinate: dict = newton_raphson.value(sensor, initial_coordinate, avg_mesured, 10**(-8), max_nr) # Newton Raphson 法で位置推定
        # converged_coordinate: dict = newton_raphson.value(sensor_origin, initial_coordinate, avg_mesured, 10**(-8), max_nr) 
        adjusted_coodinate: dict = field.value(mesured_range, converged_coordinate) # 領域外の座標調整
        if(is_cooperate == 1 and i == 0):
          sensor_origin.append(targets[i])
          sensor.append({"x": adjusted_coodinate["x"], "y": adjusted_coodinate["y"], "is_nlos": 0})
        else:
          mse += mean_squared_error.value(targets[i], adjusted_coodinate, max_loop*(all_targets - 1)) # static_target以外のMSEを算出
    rmse = math.sqrt(mse)
    total_rmse += rmse
    progress += 1
    print("\r" + "{:.3f}".format(progress/cycle*100) + "%" + " done." + " Average RMSE = " + "{:.4f}".format(total_rmse/progress), end="")
  print("\n")
  print("Average RMSE = ", total_rmse/cycle)
  print("complete.")

# 考察
# この領域内くらいだとLS法のプログラムの関数の引数がsensorとsensor_originどちらを引数にしてもほとんど変わらない
# LS_TOAにANを追加したときとIncooperateの結果は同じになる（ようにしてある）
# IncooperateもCooperateの違いは後者はTN_1に測定誤差が生じるかどうかなのでそもそも誤差の生じにくい(5.0,5.0)では結果にそこまでの差は出ないと考えられる
# 