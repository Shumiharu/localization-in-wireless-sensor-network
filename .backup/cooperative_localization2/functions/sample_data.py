import os
import sys
import numpy as np
import random
import yaml
import joblib
import pandas as pd
from datetime import datetime

# 基本関数
# from functions import distance_toa
# from functions import normalization
# from functions import line_of_position
# from functions import newton_raphson
# from functions import distance_error_squared

import distance_toa
import normalization
import line_of_position
import newton_raphson
import distance_error_squared

# 特徴量の算出
# from functions import distance_from_sensors_to_approximate_line
# from functions import distance_from_centroid_of_sensors_to_vn_maximized
# from functions import distance_from_center_of_field_to_target
# from functions import convex_hull_volume
# from functions import residual_avg
import distance_from_sensors_to_approximate_line
import distance_from_centroid_of_sensors_to_vn_maximized
import distance_from_center_of_field_to_target
import convex_hull_volume
import residual_avg

# 結果算出
# from functions import rmse_distribution
# from functions import localizable_probability_distribution
import rmse_distribution
import localizable_probability_distribution

def collect(config: dict):

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

  # Anchors & Targets Config
  anchors = config["anchors"]
  targets_count: int = config["targets"]["count"]

  # Localization Config
  # sim_cycles = config["sim_cycles"] # シミュレーション回数
  max_localization_loop = config["localization"]["max_loop"] # 最大測位回数
  channel = config["channel"] # LOSなどのチャネルを定義
  max_distance_measurement: int = config["localization"]["max_distance_measurement"] # 測距回数の最大（この回数が多いほど通信における再送回数が多くなる）
  newton_raphson_max_loop: int = config["localization"]["newton_raphson"]["max_loop"] # Newton Raphson 計算回数の最大
  newton_raphson_threshold: float = eval(config["localization"]["newton_raphson"]["threshold"]) # Newton Raphson 閾値

  # Feature 
  features_list = np.empty((0, 6))

  # Learning Model
  # model_filename = config["model"]["filename"]
  # model_filepath = "models/" + model_filename
  # model = joblib.load(model_filepath)
  model_sample = config["model"]["sample"]
  error_threshold = config["model"]["error_threshold"]
  # print(f"{model_filename} was loaded.")
  
  # Temporary Parameter
  squared_error_total = 0.0 # シミュレーション全体における合計平方根誤差
  targets_localized_count_total = 0 # シミュレーション全体における合計ターゲット測位回数
  root_mean_squared_error_list = np.array([]) # シミュレーション全体におけるRMSEのリスト
  sim_cycle = 0
  
  print("", end="\n")

  # シミュレーション開始
  while np.sum(features_list[:, -1] < error_threshold) < model_sample or np.sum(features_list[:, -1] >= error_threshold) < model_sample:
    # sensor は anchor + reference で構成
    sensors_original: np.ndarray = np.array([[anchor["x"], anchor["y"], 1] for anchor in anchors]) # 実際の座標
    sensors: np.ndarray = np.copy(sensors_original) # anchor以外は推定座標

    # ターゲット
    targets: np.ndarray = np.array([[round(random.uniform(0.0, width), 2), round(random.uniform(0.0, height), 2), 0] for target_count in range(targets_count)])

    # 平方根誤差のリスト
    squared_error_list = np.array([])

    for localization_loop in range(max_localization_loop): # unavailableの補完 本来はWhileですべてのTNが"is_localized": 1 になるようにするのがよいが計算時間短縮のため10回に設定してある（とはいってもほとんど測位されてました）
      for target in targets:
        sensors_available: np.ndarray = np.empty((0, 3))
        distances_measured: np.ndarray = np.array([])
        if target[2] == 0: # i番目のTNがまだ測位されていなければ行う
          for sensor_original, sensor in zip(sensors_original, sensors):
            distance_accurate = np.linalg.norm(target[:2] - sensor_original[:2])
            distance_measured = distance_toa.calculate(channel, max_distance_measurement, distance_accurate)
            distances_measured = np.append(distances_measured, distance_measured)
            if not np.isinf(distance_measured):
              sensors_available = np.append(sensors_available, [sensor], axis=0)
        
        # 三辺測量の条件（LOPの初期解を導出できる条件）
        distances_estimated = distances_measured[~np.isinf(distances_measured)]
        is_localizable = len(distances_estimated) >= 3
        if not is_localizable:
          continue
        
        # 測位
        target_estimated = line_of_position.calculate(sensors_available, distances_estimated) # Line of Positionによる初期解の算出
        target_estimated = newton_raphson.calculate(sensors_available, distances_estimated, target_estimated, newton_raphson_max_loop, newton_raphson_threshold) # Newton Raphson法による最適解の算出
        target_estimated = normalization.calculate(field_range, target_estimated) # 測位フィールド外に測位した場合の補正
        target_estimated = np.append(target_estimated, 0) # 測位フラグの付加
        
        if not np.any(np.isnan(target_estimated)):
        
          # 特徴量の計算
          feature_avg_residual = residual_avg.calculate(sensors_available, distances_estimated, target_estimated)
          feature_convex_hull_volume = convex_hull_volume.calculate(sensors_available)
          feature_distance_from_center_of_field_to_target = distance_from_center_of_field_to_target.calculate(field_range, target_estimated)
          feature_distance_from_centroid_of_sensors_to_vn_maximized = distance_from_centroid_of_sensors_to_vn_maximized.calculate(sensors_available, distances_estimated, target_estimated)
          feature_distance_to_approximate_line = distance_from_sensors_to_approximate_line.calculate(sensors_available)

          features = np.array([
            feature_avg_residual,
            feature_convex_hull_volume,
            feature_distance_from_center_of_field_to_target,
            feature_distance_from_centroid_of_sensors_to_vn_maximized,
            feature_distance_to_approximate_line,
          ])

          # 平均平方根誤差の算出
          squared_error = distance_error_squared.calculate(target, target_estimated)
          squared_error_list = np.append(squared_error_list, squared_error)
          
          # 誤差の特徴量はfeaturesの配列の一番最後に
          feature_error = np.sqrt(squared_error)
          features = np.append(features, feature_error)
          if feature_error < error_threshold and np.sum(features_list[:, -1] < error_threshold) < model_sample:
            features_list = np.append(features_list, [features], axis=0)
          if feature_error >= error_threshold and np.sum(features_list[:, -1] >= error_threshold) < model_sample:
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
    print(f"positive: {positive}/{model_sample} negative: {negative}/{model_sample}", end=" ")
    print("Average RMSE = " + "{:.4f}".format(root_mean_squared_error_avg) + " Average Localizable Probability = " + "{:.4f}".format(localizable_probability_avg), end="\r\r")
  print("\n")

  print(f"Average RMSE = {root_mean_squared_error_avg} m")

  sample_data = pd.DataFrame({
    "avg_residual": features_list[:, 0],
    "convex_hull_volume": features_list[:, 1], 
    "distance_from_center_of_field_to_target": features_list[:, 2],
    "distance_from_centroid_of_sensors_to_vn_maximized": features_list[:, 3],
    "distance_to_approximate_line": features_list[:, 4],
    "error": features_list[:, 5]
  })

  # RMSEの累積分布関数を出力
  # root_mean_squared_error_range = np.arange(0, 10, 0.01)
  # cumulative_distribution_function = norm.cdf(root_mean_squared_error_range, loc=np.mean(root_mean_squared_error_range), scale=np.std(root_mean_squared_error_range))
  # cumulative_distribution_function_data = pd.DataFrame({
  #   "RMSE": root_mean_squared_error_range,
  #   "CDF": cumulative_distribution_function
  # })
  # cdf_filename = "cumulative_distribution_function.csv"
  # cdf_filepath = os.path.join(output_dirpath, cdf_filename)
  # cumulative_distribution_function_data.to_csv(cdf_filepath, index=False)
  # print(f"{cdf_filename} was saved in {cdf_filepath}.")
  
  # RMSEの分布を出力
  # field_rmse_distribution_data = pd.DataFrame({
  #   "x": field_rmse_distribution[:, 0],
  #   "y": field_rmse_distribution[:, 1],
  #   "RMSE": field_rmse_distribution[:, 2],
  #   "data_count": field_rmse_distribution[:, 3],
  # })
  # field_rmse_distribution_filename = "field_rmse_distribution.csv"
  # field_rmse_distribution_filepath = os.path.join(output_dirpath, field_rmse_distribution_filename)
  # field_rmse_distribution_data.to_csv(field_rmse_distribution_filepath, index=False)
  # print(f"{field_rmse_distribution_filename} was saved in {field_rmse_distribution_filepath}.")

  # 測位可能確率の分布を出力
  # field_localizable_probability_distribution_data = pd.DataFrame({
  #   "x": field_localizable_probability_distribution[:, 0],
  #   "y": field_localizable_probability_distribution[:, 1],
  #   "localizable_probability": field_localizable_probability_distribution[:, 2],
  #   "data_count": field_localizable_probability_distribution[:, 3],
  # })
  # field_localizable_probability_distribution_filename = "field_localizable_probability_distribution.csv"
  # field_localizable_probability_distribution_filepath = os.path.join(output_dirpath, field_localizable_probability_distribution_filename)
  # field_localizable_probability_distribution_data.to_csv(field_localizable_probability_distribution_filepath, index=False)
  # print(f"{field_localizable_probability_distribution_filename} was saved in {field_localizable_probability_distribution_filepath}.")

  return sample_data

# Example Usage
if __name__ == "__main__":
  # Open configuration file
  config_filename = "config_0.yaml"
  config_filepath = "../configs/" + config_filename
  with open(config_filepath, "r") as config_file:
    config = yaml.safe_load(config_file)
    print(f"{config_filename} was loaded")

  sample_data: pd.DataFrame = collect(config)

  sample_data_filename = "sample_0.csv"
  sample_data_filepath = "../samples/" + sample_data_filename
  sample_data.to_csv(sample_data_filepath, index=False)
  print(f"{sample_data_filename} was saved in {sample_data_filepath}")