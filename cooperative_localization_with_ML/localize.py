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

import numpy as np
import random
import yaml

# 基本関数
from functions import distance_toa
from functions import field
from functions import line_of_position
from functions import newton_raphson
from functions import mean_squared_error

# 特徴量の算出
from functions import distance_from_sensors_to_approximate_line
from functions import distance_from_centroid_of_sensors_to_vn_maximized
from functions import distance_from_center_of_field_to_target
from functions import convex_hull_volume
from functions import avg_residual


if __name__ == "__main__":
  print("Coopolative Positioning Time of Arrival")

  # Open configuration file
  config_filename = "config_0.yaml"
  config_filepath = "configs/" + config_filename
  with open(config_filepath, "r") as f:
    config = yaml.safe_load(f)
    print(f"{config_filename} was loaded")

  # Cooperative Localization or not
  is_cooperative_localization = config["localization"]["is_cooperative"]
  print("Cooperative Mode" if is_cooperative_localization else "Incooperative Mode")

  # Field Config
  field_range = config["field_range"]
  width = field_range["x_top"] - field_range["x_bottom"]
  height = field_range["y_top"] - field_range["y_bottom"] 
  print("field: " + str(width) + " x " + str(height))

  # Anchors & Targets Config
  anchors = config["anchors"]
  print("anchor: (x, y) = ", end="")
  for anchor in anchors:
    anchor_x = anchor["x"]
    anchor_y = anchor["y"]
    print(f"({anchor_x}, {anchor_y})", end=" ")
  print(f"\n=> anchor count: {len(anchors)}")

  targets_count: int = config["targets"]["count"]
  print("target: (x, y) = random")
  print(f"=> target count: {targets_count}")

  # Localization Config
  sim_cycles = config["sim_cycles"] # シミュレーション回数
  max_localization_loop = config["localization"]["max_loop"] # 最大測位回数
  channel = config["channel"] # LOSなどのチャネルを定義
  max_distance_measurement: int = config["localization"]["max_distance_measurement"] # 測距回数の最大（この回数が多いほど通信における再送回数が多くなる）
  newton_raphson_max_loop: int = config["localization"]["newton_raphson"]["max_loop"] # Newton Raphson 計算回数の最大
  newton_raphson_threshold: float = eval(config["localization"]["newton_raphson"]["threshold"]) # Newton Raphson 閾値

  # Temporary Parameter
  targets_localized_count_total: int = 0
  mse_total: float = 0.0

  for sim_cycle in range(sim_cycles):
    # sensor = anchor + reference
    sensors_original: np.ndarray = np.array([[anchor["x"], anchor["y"], 1] for anchor in anchors]) # 実際の座標
    sensors: np.ndarray = np.copy(sensors_original) # anchor以外は推定座標

    # target
    targets: np.ndarray = np.array([[round(random.uniform(0.0, width), 2), round(random.uniform(0.0, height), 2), 0] for target_count in range(targets_count)])

    # 測位されたtarget
    targets_localized: np.ndarray = np.array([])

    # 平均二乗誤差
    mse = 0.0

    for localization_loop in range(max_localization_loop): # unavailableの補完 本来はWhileですべてのTNが"is_localized": 1 になるようにするのがよいが計算時間短縮のため10回に設定してある（とはいってもほとんど測位されてました）
      for target in targets:
        sensors_available: np.ndarray = np.empty((0, 3))
        distances_estimated: np.ndarray = np.array([])
        if target[2] == 0: # i番目のTNがまだ測位されていなければ行う
          for sensor_original, sensor in zip(sensors_original, sensors):
            distance_accurate = np.linalg.norm(target[:2] - sensor_original[:2])
            distance_estimated = distance_toa.value(channel, max_distance_measurement, distance_accurate)
            if not np.isinf(distance_estimated):
              sensors_available = np.append(sensors_available, [sensor], axis=0)
              distances_estimated = np.append(distances_estimated, distance_estimated) # targetとの距離
        
        # 三辺測量の条件（LOPの初期解を導出できる条件）
        if len(distances_estimated) < 3:
          continue
        
        # 測位
        target_localized = line_of_position.value(sensors_available, distances_estimated) # Line of Positionによる初期解の算出
        target_localized = newton_raphson.value(sensors_available, distances_estimated, target_localized, newton_raphson_max_loop, newton_raphson_threshold) # Newton Raphson法による最適解の算出
        target_localized = field.value(field_range, target_localized) # 測位フィールド外に測位した場合の補正
        target_localized = np.append(target_localized, 0) # 測位フラグの付加

        # 特徴量の計算
        # feature_distance_to_approximate_line: float = distance_from_sensors_to_approximate_line.value(sensors_available) # sensorとsensorから線形回帰して得られる近似直線の距離の平均
        # feature_convex_hull_volume: float = convex_hull_volume.value(sensors_available) # 凸包の面積
        # feature_avg_residual: float = avg_residual.value(sensors_available, target_localized) # 残差の平均
        # feature_distance_from_centroid_of_sensors_to_vn_maximized = distance_from_centroid_of_sensors_to_vn_maximized.value(sensors_available, target_localized, channel, max_distance_measurement) # sensorの重心とvanish nodeまでの最大距離
        # feature_distance_from_center_of_field_to_target = distance_from_center_of_field_to_target.value(field_range, target_localized) # フィールドの中心とtargetの距離
        
        # 平均二乗誤差の計算
        if not np.any(np.isnan(target_localized)):
          mse += mean_squared_error.value(target, target_localized)
        
          # 測位フラグの更新
          target[2], target_localized[2] = 1, 1

          if is_cooperative_localization:
            sensors_original = np.append(sensors_original, [target], axis=0)
            sensors = np.append(sensors, [target_localized], axis=0)
      
      targets_localized_count = np.sum(targets[:, 2] == 1)
      if targets_localized_count == len(target):
        break

    mse_total += mse
    targets_localized_count_total += targets_localized_count
    rmse = np.sqrt(mse_total/targets_localized_count_total)
    print("\r" + "{:.3f}".format((sim_cycle + 1)/sim_cycles*100) + "%" + " done." + " RMSE = " + "{:.4f}".format(rmse), end="")
  print("\n")
  print("RMSE = " + "{:.4f}".format(np.sqrt(rmse)) + " m")
print("\ncomplete.")


    # ターゲット
    # targets: array = []
    # for target_count in range(targets_count):
    # # while len(targets) < target_count:
    #   random_x = round(random.uniform(0.0, width), 2)
    #   # random_x = random.uniform(x_bottom, x_top)
    #   random_y = round(random.uniform(0.0, height), 2)
    #   # random_y = random.uniform(y_bottom, y_top)

    #   # if (random_x < x_bottom or random_x > x_top) and (random_y < y_bottom or random_y > y_top):
    #   #   targets.append({"x": random_x, "y": random_y, "is_localized": 0}) # TNを生成
    #   targets.append({"x": random_x, "y": random_y, "is_localized": 0}) # TNを生成
    #   # targets.append({"x": 1.0, "y": 1.0, "is_localized": 0}) # TNを生成（テスト）

        # センサ各位置（デフォルト以外は候補点の座標）-> sensor = sensors_original としたくなるが，そうするとsensor.appendでsensors_originalにもappendされてしまうので要注意!!
    # sensor: array = [
    #   # {"x": x_bottom, "y": y_bottom, "is_localized": 1},
    #   # {"x": x_top, "y": y_bottom, "is_localized": 1},
    #   # {"x": x_bottom, "y": y_top, "is_localized": 1},
    #   # {"x": x_top, "y": y_top, "is_localized": 1}
    # ]

            # if targets[i]["is_localized"] == 0:
        #   #  TNが測位可能か調べ，可能であれば新しくavg_measuredとsensorを構成する 
        #   # available: dict = toa_distance_limited.value(targets[i], sensors, sensors_original, channel["los"], max_distance_measurement, 0, error_distance) # センサーからの距離を測定
        #   available_avg_measured: array = available["avg_measured"]
        #   available_sensor: dict = available["sensor"]

          # TOAで測位不可能（3点以上で距離が分からない）ならいったんパスする
        #   if len(available_avg_measured) < 3: 
        #     continue
          
        #   initial_coordinate: dict = line_of_position.value(available_sensor, available_avg_measured) # Line Of Position で初期値決定
        #   converged_coordinate: dict = newton_raphson.value(available_sensor, initial_coordinate, available_avg_measured, 10**(-8), newton_raphson_max_loop) # Newton Raphson法で位置推定
            
        #   adjusted_coodinate: dict = field.value(field_range, converged_coordinate) # 領域外の座標調整
          
        #   targets_localized.append(targets[i]) # 測位済みのTNの実際の座標を追加

        #   mse = mean_squared_error.value(targets[i], adjusted_coodinate, 1) # MSEを算出
        #   total_mse += mse # RMSE算出のためにmseを加算していく
        #   total_localized_targets_count += 1 # TNの合計測位数を計算

        #   targets[i]["is_localized"] = 1 # 測位されたことを保存

        #   if is_cooperative_localization: # 協調測位の場合
        #     sensors_original.append(targets[i]) # TNの実際の座標を追加
        #     sensors.append({"x": adjusted_coodinate["x"], "y": adjusted_coodinate["y"], "is_localized": 1}) # TNの測定した候補点の座標を追加
        # else:
        #   pass