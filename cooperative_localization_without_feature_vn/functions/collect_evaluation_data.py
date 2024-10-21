import os
import sys
import numpy as np
import random
import yaml
import joblib
import pandas as pd
from datetime import datetime

# 基本関数
from basis import distance_toa
from basis import normalization
from basis import line_of_position
from basis import newton_raphson

# 特徴量の算出
from feature import convex_hull_volume
from feature import distance_from_sensors_to_approximate_line
from feature import distance_from_center_of_field_to_target
from feature import distance_from_centroid_of_sn_available_to_tn_estimated
from feature import residual_avg
from feature import distance_error_squared

# 結果算出
from result import rmse_distribution
from result import localizable_probability_distribution


if __name__ == "__main__":
  
  args = sys.argv
  is_subprocess = True if len(args) == 2 else False

  # Open configuration file
  config_filename = "config_0.yaml"
  config_filepath = "../configs/" + config_filename
  if is_subprocess:
    config_filepath = os.path.join(args[1], "config.yaml")
  with open(config_filepath, "r") as config_file:
    config = yaml.safe_load(config_file)
    print(f"{config_filename} was loaded from {config_filepath}")

  # Cooperative Localization or not
  is_cooperative_localization = config["localization"]["is_cooperative"]
  print("Localization: Least Square (LS) Method", end=" ")
  print("with Cooperation" if is_cooperative_localization else "without Cooperation")

  # Field Config
  field_range = config["field_range"]
  
  grid_interval = field_range["grid_interval"]
  x_range = np.arange(field_range["x_bottom"], field_range["x_top"] + grid_interval, grid_interval)
  y_range = np.arange(field_range["y_bottom"], field_range["y_top"] + grid_interval, grid_interval)
  field_rmse_distribution = np.array([[x, y, 0.0, 0] for x in x_range for y in y_range]) # rmseの分布を算出 -> [x, y, rmse, data_count]
  field_localizable_probability_distribution = np.copy(field_rmse_distribution) # 測位可能確立の分布を算出 -> [x, y, localizable probability, data_count]

  width = field_range["x_top"] - field_range["x_bottom"]
  height = field_range["y_top"] - field_range["y_bottom"]
  print(f"field: {width} x {height}")

  # Anchors & Targets Config
  anchors_config = config["anchors"]
  print("anchor: (x, y) = ", end="")
  for anchor_config in anchors_config:
    anchor_x = anchor_config["x"]
    anchor_y = anchor_config["y"]
    print(f"({anchor_x}, {anchor_y})", end=" ")
  print(f"\n=> anchor count: {len(anchors_config)}")

  targets_count: int = config["targets"]["count"]
  print("target: (x, y) = random")
  print(f"=> target count: {targets_count}", end="\n\n")

  # Localization Config
  max_localization_loop = config["localization"]["max_loop"] # 最大測位回数
  channel = config["channel"] # LOSなどのチャネルを定義
  max_distance_measurement: int = config["localization"]["max_distance_measurement"] # 測距回数の最大（この回数が多いほど通信における再送回数が多くなる）
  newton_raphson_max_loop: int = config["localization"]["newton_raphson"]["max_loop"] # Newton Raphson 計算回数の最大
  newton_raphson_threshold: float = eval(config["localization"]["newton_raphson"]["threshold"]) # Newton Raphson 閾値

  # Feature 
  features_list = np.empty((0, 5))

  # Evaluation
  evaluation_count = config["evaluation_data"]["count"]
  evaluation_data_filename = config["evaluation_data"]["filename"]
  evaluation_data_filepath = "../evaluation_data/" + evaluation_data_filename
  print(f"{evaluation_data_filename} will be saved in {evaluation_data_filepath}.")

  # Model
  error_threshold = config["model"]["error_threshold"]
  
  # Temporary Parameter
  squared_error_total = 0.0 # シミュレーション全体における合計平方根誤差
  targets_localized_count_total = 0 # シミュレーション全体における合計ターゲット測位回数
  root_mean_squared_error_list = np.array([]) # シミュレーション全体におけるRMSEのリスト
  sim_cycle = 0
  
  print("\n")

  # シミュレーション開始
  while np.sum(features_list[:, -1] < error_threshold) < evaluation_count or np.sum(features_list[:, -1] >= error_threshold) < evaluation_count:
    # sensor は anchor node と reference node で構成
    sensors_original: np.ndarray = np.array([[anchor_config["x"], anchor_config["y"], 1] for anchor_config in anchors_config]) # 実際の座標
    sensors: np.ndarray = np.copy(sensors_original) # anchor以外は推定座標

    # ターゲット
    targets: np.ndarray = np.array([[round(random.uniform(0.0, width), 2), round(random.uniform(0.0, height), 2), 0] for target_count in range(targets_count)])

    # 平方根誤差のリスト
    squared_error_list = np.array([])
    
    # 測距最大距離
    distance_measured_max = 0.0

    for localization_loop in range(max_localization_loop): # unavailableの補完 本来はWhileですべてのTNが"is_localized": 1 になるようにするのがよいが計算時間短縮のため10回に設定してある（とはいってもほとんど測位されてました）
      for target in targets:
        distances_measured: np.ndarray = np.array([])
        if target[2] == 0: # TNがまだ測位されていなければ行う
          for sensor_original, sensor in zip(sensors_original, sensors):
            distance_accurate = np.linalg.norm(target[:2] - sensor_original[:2])
            distance_measured = distance_toa.calculate(channel, max_distance_measurement, distance_accurate)
            distances_measured = np.append(distances_measured, distance_measured)

        # 三辺測量の条件（LOPの初期解を導出できる条件）
        distances_estimated = distances_measured[~np.isinf(distances_measured)]
        is_localizable = len(distances_estimated) >= 3
        if not is_localizable:
          continue
        
        # 測距最大距離の更新
        distance_measured_max = max(distance_measured_max, np.max(distances_measured))

        # 測位可能なセンサ
        sensors_available = sensors[~np.isinf(distances_measured)]

        # 測位
        target_estimated = line_of_position.calculate(sensors_available, distances_estimated) # Line of Positionによる初期解の算出
        target_estimated = newton_raphson.calculate(sensors_available, distances_estimated, target_estimated, newton_raphson_max_loop, newton_raphson_threshold) # Newton Raphson法による最適解の算出
        target_estimated = normalization.calculate(field_range, target_estimated) # 測位フィールド外に測位した場合の補正
        target_estimated = np.append(target_estimated, 0) # 測位フラグの付加
        
        if not np.any(np.isnan(target_estimated)):
        
          # 特徴量の計算
          feature_convex_hull_volume = convex_hull_volume.calculate(sensors_available)
          feature_distance_from_center_of_field_to_target = distance_from_center_of_field_to_target.calculate(field_range, target_estimated)
          # feature_distance_from_centroid_of_sn_available_to_tn_estimated = distance_from_centroid_of_sn_available_to_tn_estimated.calculate(sensors_available, target_estimated)
          feature_distance_from_sensors_to_approximate_line = distance_from_sensors_to_approximate_line.calculate(sensors_available)
          feature_residual_avg = residual_avg.calculate(sensors_available, distances_estimated, target_estimated)

          features = np.array([
            feature_convex_hull_volume,
            feature_distance_from_center_of_field_to_target,
            # feature_distance_from_centroid_of_sn_available_to_tn_estimated,
            feature_distance_from_sensors_to_approximate_line,
            feature_residual_avg,
          ])

          # 平均平方根誤差の算出
          squared_error = distance_error_squared.calculate(target, target_estimated)
          squared_error_list = np.append(squared_error_list, squared_error)
          
          # 誤差の特徴量はfeaturesの配列の一番最後に
          feature_error = np.sqrt(squared_error)
          features = np.append(features, feature_error)
          if feature_error < error_threshold and np.sum(features_list[:, -1] < error_threshold) < evaluation_count:
            features_list = np.append(features_list, [features], axis=0)
          if feature_error >= error_threshold and np.sum(features_list[:, -1] >= error_threshold) < evaluation_count:
            features_list = np.append(features_list, [features], axis=0)

          # 測位フラグの更新
          target[2], target_estimated[2] = 1, 1
          targets_localized = targets[targets[:, 2] == 1]
          if len(targets_localized) == targets_count:
            break

          if is_cooperative_localization:
            sensors_original = np.append(sensors_original, [target], axis=0)
            sensors = np.append(sensors, [target_estimated], axis=0)
      else:
        continue
      break
    
    # シミュレーション全体におけるMSE及びRMSEの算出
    squared_error_total += np.sum(squared_error_list)
    targets_localized_count_total += len(targets_localized)
    mean_squared_error = squared_error_total/targets_localized_count_total
    root_mean_squared_error = np.sqrt(mean_squared_error)
      
    # 求めたRMSEをリストに追加
    root_mean_squared_error_list = np.append(root_mean_squared_error_list, root_mean_squared_error)

    # 平均のRMSEの算出
    if sim_cycle == 0:
      root_mean_squared_error_avg = root_mean_squared_error
    else:
      root_mean_squared_error_avg = (root_mean_squared_error_avg*sim_cycle + root_mean_squared_error)/(sim_cycle + 1)
    
    # RMSEの分布を更新（協調測位の場合はRMSEの値が大きく振れるのであまり意味がないかも）
    # field_rmse_distribution = rmse_distribution.update(field_rmse_distribution, grid_interval, targets_localized, squared_error_list)

    # 測位可能確率の分布の更新とその平均の算出
    field_localizable_probability_distribution = localizable_probability_distribution.update(field_localizable_probability_distribution, grid_interval, targets, targets_localized)
    localizable_probability_avg = np.sum(field_localizable_probability_distribution[:, 2]*field_localizable_probability_distribution[:, 3])/np.sum(field_localizable_probability_distribution[:, 3])

    sim_cycle += 1
    positive = np.sum(features_list[:, -1] < error_threshold)
    negative = np.sum(features_list[:, -1] >= error_threshold)
    print(f"positive: {positive}/{evaluation_count} negative: {negative}/{evaluation_count}", end=" ")
    print("Average RMSE = " + "{:.4f}".format(root_mean_squared_error_avg) + " Average Localizable Probability = " + "{:.4f}".format(localizable_probability_avg), end="\r\r")
  print("\n")

  print(f"Average RMSE = {root_mean_squared_error_avg} m")

  features_data = pd.DataFrame({
    "convex_hull_volume": features_list[:, 0], 
    "distance_from_center_of_field_to_target": features_list[:, 1],
    # "distance_from_centroid_of_sn_available_to_tn_estimated": features_list[:, 2],
    "distance_from_sensors_to_approximate_line": features_list[:, 2],
    "residual_avg": features_list[:, 3],
    "error": features_list[:, 4]
  })

  features_data.to_csv(evaluation_data_filepath, index=False)
  print(f"{evaluation_data_filename} was saved in {evaluation_data_filepath}")