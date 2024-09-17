# 全てのTNが協調する測位のプログラム

#       10m
# ○-------------○
# |  ○    ○     |
# |        ○    |
# |     ○       | 10m
# |  ○        ○ |
# |        ○    |
# ○-------------○
# 
# LOS環境
# TN数: 可変
# TNは同時にランダム生成
# TN_1を測位したあとANにしてTN_2を測位（協調測位）
# 全てのTNに対するRMSEを算出する

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
  print("Coopolative Positioning Time of Arrival")
  all_targets: int = 20 # ターゲットの個数 all_targets = 1 はLS_TOA.pyと同じ
  print("TN: " + str(all_targets) + " dynamic")

  # 協調しているか
  is_cooperate: int = 1
  print("Cooperative Mode" if is_cooperate == 1 else "Incooperative Mode")

  # LOPとNewton RaphsonでTNを実際の座標で使うか推定の座標で使うか
  is_original: int = 1
  print("use sensor_original" if is_original == 1 else "use sensor")

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
  cycle = 1000

  for h in range(cycle):
    mse: float = 0.0

    # ターゲット
    targets: array = []
    for i in range(all_targets):
      targets.append({"x": (float(random.randrange(101)/10)), "y": (float(random.randrange(101)/10)), "is_nlos": 0}) # TNを生成
      # targets.append({"x": 1.0, "y": 1.0, "is_nlos": 0}) # TNを生成（テスト）
      # is_nlos: float = random.randrange(100)%2
      # targets.append({"x": (float(random.randrange(101)/10)), "y": (float(random.randrange(101)/10)), "is_nlos": is_nlos})

    for loop in range(max_loop):
      # センサ各位置（実際の位置）
      sensor_origin: array = [
        {"x": x_bottom, "y": y_bottom, "is_nlos": 0},
        {"x": x_top, "y": y_bottom, "is_nlos": 0},
        {"x": x_bottom, "y": y_top, "is_nlos": 0},
        {"x": x_top, "y": y_top, "is_nlos": 0}
      ]

      # センサ各位置（デフォルト以外は候補点の座標）-> sensor = sensor_origin としたくなるが，そうするとsensor.appendでsensor_originにもappendされてしまうので要注意!!
      sensor: array = [
        {"x": x_bottom, "y": y_bottom, "is_nlos": 0},
        {"x": x_top, "y": y_bottom, "is_nlos": 0},
        {"x": x_bottom, "y": y_top, "is_nlos": 0},
        {"x": x_top, "y": y_top, "is_nlos": 0}
      ]

      for i in range(len(targets)):
        avg_mesured = toa_mesurement.value(targets[i], sensor_origin, los, nlos, max_test) # センサーからの距離を測定
        if(is_original):
          initial_coordinate: dict = line_of_position.value(sensor_origin, avg_mesured)
          converged_coordinate: dict = newton_raphson.value(sensor_origin, initial_coordinate, avg_mesured, 10**(-8), max_nr) 
        else:
          initial_coordinate: dict = line_of_position.value(sensor, avg_mesured) # Line Of Position で初期値決定
          converged_coordinate: dict = newton_raphson.value(sensor, initial_coordinate, avg_mesured, 10**(-8), max_nr) # Newton Raphson法で位置推定
        adjusted_coodinate: dict = field.value(mesured_range, converged_coordinate) # 領域外の座標調整
        mse += mean_squared_error.value(targets[i], adjusted_coodinate, max_loop*all_targets) # MSEを算出
        if(is_cooperate):
          sensor_origin.append(targets[i]) # TNの実際の座標を追加
          sensor.append({"x": adjusted_coodinate["x"], "y": adjusted_coodinate["y"], "is_nlos": 0}) # TNの測定した候補点の座標を追加
    rmse = math.sqrt(mse)
    total_rmse += rmse
    progress += 1
    print("\r" + "{:.3f}".format(progress/cycle*100) + "%" + " done." + " Average RMSE = " + "{:.4f}".format(total_rmse/progress), end="")
  print("\n")
  print("Average RMSE = ", total_rmse/cycle)
  print("complete.")

  # 考察
  # Cooperateの方でRMSEが良くなったが，-> TN: 20のとき，0.29110914592886966 m
  # LS法のプログラムの関数の引数がsensorとsensor_originで大きく結果が変わった -> TN: 20のとき，sensor: 0.29110914592886966 m, sensor_origin: 0.37754910566039784 m
  # sensorが引数の方が圧倒的にRMSEが良い．なぜか? -> newton_raphsonの関数にsensor_originを入れると悪くなった（要検討）
