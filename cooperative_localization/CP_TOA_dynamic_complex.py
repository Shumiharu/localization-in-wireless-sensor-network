# 全てのTNが協調する測位のプログラム

#       10m
# ○-------------○
# | ● ->      ○ |
# |  ○       ○  |
# |     ○       | 10m
# | ○       ○   |
# |    ○      ○ |
# ○-------------○
# 
# LOS環境
# TN数: 可変
# TNはmain_targetとしてひとつとり，その他はランダムに生成する
# ランダムに協調測位を行う．main_targetの測位が終わった時点でその処理は終了
# 
# 全てTNに対するRMSEを算出する

from array import array
import os
import csv
import numpy as np
import math 
import random

from functions import toa_mesurement
from functions import field
from functions import line_of_position
from functions import newton_raphson
from functions import mean_squared_error

if __name__ == "__main__":
  # 出力ファイルの初期化
  file: str = os.path.basename(__file__)
  file_name: str = os.path.splitext(file)[0]
  output_file: str = "./output/" + file_name + ".csv"
  output_file_name: str = file_name + ".csv"
  
  with open(output_file, "w") as f:
    pass
  print(output_file_name + " was initialized.")

  is_cooperate: int = 1
  print("Cooperative Mode" if is_cooperate == 1 else "Incooperative Mode")

  all_targets: int = 20 # ターゲットの個数 all_targets = 1 はLS_TOA.pyと同じ
  print("TN: " + str(all_targets) + " dynamic")

  # LOPとNewton RaphsonでTNを実際の座標で使うか推定の座標で使うか
  is_original: int = 0
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
  measured_range : dict = {
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

  total_rmse: float = 0.0

  progress: int = 0

  for x_grid in range(0, 101, 1):
    for y_grid in range(0, 101, 1):
      mse: float = 0.0

      # ターゲット
      main_target: dict = {"x": (float(x_grid/10)), "y": (float(y_grid/10))}
      targets: array = []
      for i in range(all_targets):
        if(i == 0):
          targets.append(main_target)
        else:
          targets.append({"x": (float(random.randrange(101)/10)), "y": (float(random.randrange(101)/10)), "is_nlos": 0})
          # is_nlos: int = random.randrange(100)%2
          # targets.append({"x": (float(random.randrange(101)/10)), "y": (float(random.randrange(101)/10)), "is_nlos": is_nlos})
      random.shuffle(targets)

      for loop in range(max_loop):
        # センサ各位置
        sensor_origin: array = [
          {"x": x_bottom, "y": y_bottom, "is_nlos": 0.0},
          {"x": x_top, "y": y_bottom, "is_nlos": 0.0},
          {"x": x_bottom, "y": y_top, "is_nlos": 0.0},
          {"x": x_top, "y": y_top, "is_nlos": 0.0}
        ]

         # センサ各位置（デフォルト以外は候補点の座標）-> sensor = sensor_origin としたくなるが，そうするとsensor.appendでsensor_originにもappendされてしまうので要注意!!
        sensor: array = [
          {"x": x_bottom, "y": y_bottom, "is_nlos": 0},
          {"x": x_top, "y": y_bottom, "is_nlos": 0},
          {"x": x_bottom, "y": y_top, "is_nlos": 0},
          {"x": x_top, "y": y_top, "is_nlos": 0}
        ]

        for i in range(len(targets)):
          avg_measured = toa_mesurement.value(targets[i], sensor_origin, los, nlos, max_test) # センサーからの距離を測定
          if(is_original):
            initial_coordinate: dict = line_of_position.value(sensor_origin, avg_measured)
            # converged_coordinate: dict = newton_raphson.value(sensor_origin, initial_coordinate, avg_measured, 10**(-8), max_nr) 
          else:
            initial_coordinate: dict = line_of_position.value(sensor, avg_measured) # Line Of Position で初期値決定
            # converged_coordinate: dict = newton_raphson.value(sensor, initial_coordinate, avg_measured, 10**(-8), max_nr) # Newton Raphson法で位置推定
          # adjusted_coordinate: dict = field.value(measured_range, converged_coordinate) # 領域外の座標調整
          adjusted_coordinate: dict = field.value(measured_range, initial_coordinate) # 領域外の座標調整
          # main_tagetにあたったら処理を終了する
          if(targets[i] == main_target):
            mse += mean_squared_error.value(targets[i], adjusted_coordinate, max_loop) # MSEを算出
            # print(f"accurate_coordinate: (x, y) = ({main_target["x"]}, {main_target["y"]})")
            # print(f"initial_coordinate: (x, y) = ({initial_coordinate["x"]}, {initial_coordinate["y"]})")
            a = math.sqrt((main_target["x"] - initial_coordinate["x"])**2 + (main_target["y"] - initial_coordinate["y"])**2)
            if a > 1.0:
              print("初期解と実際の座標との差が大きい！")
            break
          elif(targets[i]["is_nlos"] == 0.0): # losならTNをANとして追加
            sensor_origin.append(targets[i])
            sensor.append({"x": adjusted_coordinate["x"], "y": adjusted_coordinate["y"], "is_nlos": 0.0})
      rmse = math.sqrt(mse) # RMSEを算出
      output = [np.squeeze(main_target["x"]), np.squeeze(main_target["y"]), rmse] 
      with open(output_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow(output)
      total_rmse += rmse
      progress += 1
      print("\r" + "{:.3f}".format(progress/(101*101)*100) + "%" + " done." + " Average RMSE = " + "{:.4f}".format(total_rmse/progress), end="")
  print("\n")
  print("Average RMSE = ", total_rmse/(101.0*101.0))
  print("complete.")

  # 1. Average RMSE =  0.2723967253689642 推定センサ座標  w/ newton raphson -> 3. から推定センサ座標を使った場合，NewtonRaphson使った方が良い -> 初期解が適切に得られている．
  # 2. Average RMSE =  0.3602324036934757 確定センサ座標  w/ newton raphson -> 4から局所最適解を選んでしまい特性が劣化している可能性あり
  # 3. Average RMSE =  0.31470331167956367 推定センサ座標  w/o newton raphson
  # 4. Average RMSE =  0.35763089751899696 確定センサ座標  w/o newton raphson
  # 以上の結果から，確定センサ座標はLOPによる初期解の値がよくないため，Newton Raphson法により局所最適解を導出してしまう可能性がある．
