# 全てのTNが協調する測位のプログラム

#       30m
# +------------+
# |  ○    ○    |
# |   ●    ●   |
# |○    ○      | 30m
# |   ●    ● ○ |
# |        ○   |
# +------------+
# 
# ○: TN
# ●: default AN 
# 
# LOS環境
# TN数: 可変(20)
# TNは同時にランダム生成
# TN_1が測位範囲外（limited_toa_measurement.pyに記載）ならいったんパスしてTN_2を測位する
# 測位できたものから順にMSEを測定し．TNをANに追加する（協調）
# これを繰り返しすべてのTNを測定する．

from array import array
import math 
import random

from functions import limited_toa_measurement
from functions import field
from functions import line_of_position
from functions import newton_raphson
from functions import mean_squared_error

if __name__ == "__main__":
  print("Coopolative Positioning Time of Arrival")

  all_targets: int = 20 # ターゲットの個数 all_targets = 1 はLS_TOA.pyと同じ
  print("TN: " + str(all_targets) + " ")

  # 協調しているか
  is_cooperate: int = 1
  print("Cooperative Mode" if is_cooperate == 1 else "Incooperative Mode")

  max_test: int = 1 # 距離の測定回数の最大．この回数が多いほど通信における再送回数が多くなる
  max_loop: int = 1 # 試行回数の最大（1）
  max_nr: int = 10 # Newton Raphson 計算回数の最大

  # 領域の1辺の長さ
  width: float = 30.0
  height: float = 30.0

  # 測定範囲
  measured_range : dict = {
    "x_top": width,
    "x_bottom": 0.0,
    "y_top": height,
    "y_bottom": 0.0
  }

  print("field: " + str(width) + " x " + str(height))

  # ANの座標定義
  x_top: float = (width/3.0)*2.0
  x_bottom: float = width/3.0
  y_top: float = (height/3.0)*2.0
  y_bottom: float = height/3.0

  print("default anchor node: (" + str(x_bottom) + "," + str(y_bottom) + "),(" + str(x_top) + "," + str(y_bottom) + "),(" + str(x_top) + "," + str(y_top) + "),(" + str(x_bottom) + "," + str(y_top) + ")\n")

  # 測定可能最大距離 -> ANの対角の距離
  max_distance: float = math.sqrt((x_top - x_bottom)**2 + (y_top - y_bottom)**2)

  # エラー距離
  error_distance: float = 2**16

  # LOSとNLOS環境におけるそれぞれの分散と平均とノード
  los: dict = {
    "sigma": ((2.0*(10**(-9)))*(3.0*(10**(8)))), # 正規化距離誤差
    "avg": 0.0, # 
  }

  # nlosは除外される
  # nlos: dict = {
  #   "sigma": 0.809, # 正規化距離誤差
  #   "avg": 1.62, 
  # }

  # 測位に成功したTNの総和
  total_localized_targets_count: int = 0

  # 合計MSE
  total_mse: float = 0.0

  # プログラム施行回数
  max_cycle = 10000

  # 進捗
  progress: int = 0

  for cycle in range(max_cycle):
    # 測位に成功したTN
    localized_targets: array = []

    # ターゲット
    targets: array = []
    # for target in range(all_targets):
    while len(targets) < all_targets:
      random_x = random.uniform(0.0, width)
      # random_x = random.uniform(x_bottom, x_top)
      random_y = random.uniform(0.0, height)
      # random_y = random.uniform(y_bottom, y_top)

      # if (random_x < x_bottom or random_x > x_top) and (random_y < y_bottom or random_y > y_top):
      #   targets.append({"x": random_x, "y": random_y, "is_localized": 0}) # TNを生成
      targets.append({"x": random_x, "y": random_y, "is_localized": 0}) # TNを生成
      # targets.append({"x": 1.0, "y": 1.0, "is_localized": 0}) # TNを生成（テスト）

    # for loop in range(max_loop): -> 1なので省略
    # センサ各位置（実際の位置）
    sensor_origin: array = [
      {"x": x_bottom, "y": y_bottom, "is_localized": 1},
      {"x": x_top, "y": y_bottom, "is_localized": 1},
      {"x": x_bottom, "y": y_top, "is_localized": 1},
      {"x": x_top, "y": y_top, "is_localized": 1}
    ]

    # センサ各位置（デフォルト以外は候補点の座標）-> sensor = sensor_origin としたくなるが，そうするとsensor.appendでsensor_originにもappendされてしまうので要注意!!
    sensor: array = [
      {"x": x_bottom, "y": y_bottom, "is_localized": 1},
      {"x": x_top, "y": y_bottom, "is_localized": 1},
      {"x": x_bottom, "y": y_top, "is_localized": 1},
      {"x": x_top, "y": y_top, "is_localized": 1}
    ]

    for h in range(10): # unavailableの補完 本来はWhileですべてのTNが"is_localized": 1 になるようにするのがよいが計算時間短縮のため10回に設定してある（とはいってもほとんど測位されてました）
      for i in range(len(targets)):
        # i番目のTNがまだ測位されていなければ行う
        if targets[i]["is_localized"] == 0:
          #  TNが測位可能か調べ，可能であれば新しくavg_measuredとsensorを構成する 
          available: dict = limited_toa_measurement.value(targets[i], sensor, sensor_origin, los, max_test, max_distance, error_distance) # センサーからの距離を測定
          available_avg_measured: array = available["avg_measured"]
          available_sensor: dict = available["sensor"]

          # TOAで測位不可能（3点以上で距離が分からない）ならいったんパスする
          if len(available_avg_measured) < 3: 
            continue
          
          initial_coordinate: dict = line_of_position.value(available_sensor, available_avg_measured) # Line Of Position で初期値決定
          converged_coordinate: dict = newton_raphson.value(available_sensor, initial_coordinate, available_avg_measured, 10**(-8), max_nr) # Newton Raphson法で位置推定
            
          adjusted_coodinate: dict = field.value(measured_range, converged_coordinate) # 領域外の座標調整
          
          localized_targets.append(targets[i]) # 測位済みのTNの実際の座標を追加

          mse = mean_squared_error.value(targets[i], adjusted_coodinate, max_loop) # MSEを算出
          total_mse += mse # RMSE算出のためにmseを加算していく
          total_localized_targets_count += 1 # TNの合計測位数を計算

          targets[i]["is_localized"] = 1 # 測位されたことを保存

          if is_cooperate: # 協調測位の場合
            sensor_origin.append(targets[i]) # TNの実際の座標を追加
            sensor.append({"x": adjusted_coodinate["x"], "y": adjusted_coodinate["y"], "is_localized": 1}) # TNの測定した候補点の座標を追加
        else:
          pass
    progress += 1
    # RMSEと測位成功率（とりあえず測位できたということあってるかどうかはおいておく）を出力
    print("\r" + "{:.3f}".format(progress/max_cycle*100) + "%" + " done." + " RMSE = " + "{:.4f}".format(math.sqrt(total_mse/total_localized_targets_count)) + " Positioning success rate: " + "{:.4f}".format(total_localized_targets_count/(all_targets*progress)*100) + "%", end="")
  print("\n")
  print("RMSE = " + "{:.4f}".format(math.sqrt(total_mse/total_localized_targets_count)) + " m")
  print("Positioning success rate = " + "{:.4f}".format(total_localized_targets_count/(all_targets*progress)*100) + " %"), 
print("\ncomplete.")