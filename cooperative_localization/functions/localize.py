import os
import sys
import numpy as np
import random
import yaml
import joblib
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 基本関数
from basis import distance_toa
from basis import normalization
from basis import line_of_position
from basis import newton_raphson

# 特徴量の算出
from feature import distance_from_sensors_to_approximate_line
from feature import distance_from_center_of_field_to_target
from feature import convex_hull_volume
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

  # Learning Model
  is_predictive = config["localization"]["is_predictive"]
  is_recursive = config["localization"]["is_recursive"]
  if is_predictive:
    error_threshold = config["model"]["error_threshold"]
    model_filename = config["model"]["filename"]
    model_filepath = "../models/" + model_filename
    model = joblib.load(model_filepath)
    print("Error 'Recursive' Prediction" if is_recursive else "Error Prediction", end=" ")
    print(f"by Machine Learning (model: {model_filename})")
  else:
    print("No Error Prediction")

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
  sim_cycles = config["sim_cycles"] # シミュレーション回数
  max_localization_loop = config["localization"]["max_loop"] # 最大測位回数
  channel = config["channel"] # LOSなどのチャネルを定義
  max_distance_measurement: int = config["localization"]["max_distance_measurement"] # 測距回数の最大（この回数が多いほど通信における再送回数が多くなる）
  newton_raphson_max_loop: int = config["localization"]["newton_raphson"]["max_loop"] # Newton Raphson 計算回数の最大
  newton_raphson_threshold: float = eval(config["localization"]["newton_raphson"]["threshold"]) # Newton Raphson 閾値

  # Temporary Parameter
  squared_error_total = 0.0 # シミュレーション全体における合計平方根誤差
  targets_localized_count_total = 0 # シミュレーション全体における合計ターゲット測位回数
  root_mean_squared_error_list = np.array([]) # シミュレーション全体におけるRMSEのリスト
  # squared_error_lists = np.empty((0,20))

  # Make Folder and Save Config
  if is_subprocess:
    output_dirpath = args[1]
  else:
    now = datetime.now()
    output_dirname = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dirpath = "../output/" + output_dirname
    os.makedirs(output_dirpath, exist_ok=True)
    print(f"{output_dirname} was made.")

    config_saved_filepath = os.path.join(output_dirpath, 'config.yaml')
    with open(config_saved_filepath, "w") as config_saved_file:
      yaml.safe_dump(config, config_saved_file)
      print(f"{config_filename} was saved.")
  
  print("", end="\n")

  # シミュレーション開始
  for sim_cycle in range(sim_cycles):
    # sensor は anchor と reference で構成
    anchors = np.array([[anchor_config["x"], anchor_config["y"], 1] for anchor_config in anchors_config])
    sensors_original = np.copy(anchors) # 実際の座標
    sensors: np.ndarray = np.copy(sensors_original) # anchor以外は推定座標

    # ターゲット
    targets: np.ndarray = np.array([[round(random.uniform(0.0, width), 2), round(random.uniform(0.0, height), 2), 0] for target_count in range(targets_count)])

    # 平方根誤差のリスト
    squared_error_list = np.array([])
    # squared_error_list = np.array([np.nan]*targets_count)

    for localization_loop in range(max_localization_loop):
      for target in targets:
        distances_measured: np.ndarray = np.array([]) # 測距値（測距不可でも代入）
        if target[2] == 0: # i番目のTNがまだ測位されていなければ行う
          for sensor_original, sensor in zip(sensors_original, sensors):
            distance_accurate = np.linalg.norm(target[:2] - sensor_original[:2])
            distance_measured = distance_toa.calculate(channel, max_distance_measurement, distance_accurate)
            distances_measured = np.append(distances_measured, distance_measured)
        
        # 三辺測量の条件（LOPの初期解を導出できる条件）
        mask_distance_measurable = ~np.isinf(distances_measured)
        distances_estimated = distances_measured[mask_distance_measurable]
        sensors_available = sensors[mask_distance_measurable]
        sensors_available_initial = np.copy(sensors_available)

        is_initial_judge = True
        target_estimated_mean = np.zeros(len(target))
        recursive_count = 0

        while len(distances_estimated) >= 3:

          # 測位
          target_estimated = line_of_position.calculate(sensors_available, distances_estimated) # Line of Positionによる初期解の算出
          target_estimated = newton_raphson.calculate(sensors_available, distances_estimated, target_estimated, newton_raphson_max_loop, newton_raphson_threshold) # Newton Raphson法による最適解の算出
          target_estimated = normalization.calculate(field_range, target_estimated) # 測位フィールド外に測位した場合の補正
          target_estimated = np.append(target_estimated, 0) # 測位フラグの付加
          
          if not np.any(np.isnan(target_estimated)):
            
            if is_predictive:

              # 特徴量の計算
              feature_convex_hull_volume = convex_hull_volume.calculate(sensors_available)
              feature_distance_from_center_of_field_to_target = distance_from_center_of_field_to_target.calculate(field_range, target_estimated)
              feature_distance_from_sensors_to_approximate_line = distance_from_sensors_to_approximate_line.calculate(sensors_available)
              feature_residual_avg = residual_avg.calculate(sensors_available, distances_estimated, target_estimated)

              features = np.array([
                feature_convex_hull_volume,
                feature_distance_from_center_of_field_to_target,
                feature_distance_from_sensors_to_approximate_line,
                feature_residual_avg,
              ]) 

            if not is_predictive or not model.predict([features]):
              
              # 平均平方根誤差の算出
              squared_error = distance_error_squared.calculate(target, target_estimated)
              squared_error_list = np.append(squared_error_list, squared_error)
              # order_localized = len(targets_localized) - 1
              # squared_error_list[order_localized] = squared_error

              # 測位フラグの更新
              target[2], target_estimated[2] = 1, 1

              # 協調測位の場合はReference Nodeとしてセンサを追加する
              if is_cooperative_localization:
                sensors_original = np.append(sensors_original, [target], axis=0)
                sensors = np.append(sensors, [target_estimated], axis=0)

              break

            else:
              if is_recursive:
                if is_initial_judge or np.linalg.norm(target_estimated[:2] - target_estimated_previous[:2]) < error_threshold:
                  in_anchors = np.array([any(np.all(sensor_available == anchors, axis=1)) for sensor_available in sensors_available])
                  
                  if np.all(in_anchors) and recursive_count == recursive_count_max: # ここのif文をis_initial_judgeにすると測位確率は99.99%になるが測位精度が大きく劣化

                    # 平均平方根誤差の算出
                    squared_error = distance_error_squared.calculate(target, target_estimated_mean)
                    squared_error_list = np.append(squared_error_list, squared_error)

                    # 測位フラグの更新
                    target[2], target_estimated_mean[2] = 1, 1

                    # 協調測位の場合はReference Nodeとしてセンサを追加する
                    if is_cooperative_localization:
                      sensors_original = np.append(sensors_original, [target], axis=0)
                      sensors = np.append(sensors, [target_estimated_mean], axis=0)

                    break

                  if not np.all(in_anchors):
                    if is_initial_judge:
                      is_initial_judge = False
                      recursive_count_max = len(sensors_available[~in_anchors])

                    mask_rn = np.where(~in_anchors)[0]
                    distances_estimated_from_rn = distances_estimated[mask_rn]
                    mask_rn_max = mask_rn[np.argmax(distances_estimated_from_rn)]

                    distances_estimated = np.delete(distances_estimated, mask_rn_max)
                    sensors_available = np.delete(sensors_available, mask_rn_max, axis=0)

                    target_estimated_previous = target_estimated
                    target_estimated_mean = (target_estimated_mean*recursive_count + target_estimated)/(recursive_count + 1)

                    recursive_count += 1
                  else:
                    break
                else:
                  break
              else:
                break
          else:
            break

        targets_localized = targets[targets[:, 2] == 1] # 推定座標ではないので注意
        if len(targets_localized) == targets_count:
          break
      else:
        continue
      break
    

    # targets_not_localized = targets[targets[:, 2] == 0]
    # if len(targets_not_localized) > 0 and np.all((13.0 < target[:2]) & (target[:2] < 17.0)):
    #   print(f"再帰回数: {recursive_count}")
    #   plt.scatter(sensors[:, 0], sensors[:, 1], c="gray")
    #   plt.scatter(sensors_available_initial[:, 0], sensors_available_initial[:, 1], c="black")
    #   plt.scatter(sensors_available[:, 0], sensors_available[:, 1], c="green")
    #   plt.scatter(target_estimated_previous[0], target_estimated_previous[1], c="orange")
    #   plt.scatter(targets_not_localized[:, 0], targets_not_localized[:, 1], c="blue")
    #   plt.scatter(target[0], target[1], c="red")
    #   plt.show()
    #   plt.close('all')
    #   plt.clf()
    
    # シミュレーション全体におけるMSE及びRMSEの算出
    squared_error_total += np.sum(squared_error_list)
    # squared_error_total += np.nansum(squared_error_list)
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

    # 測位順と測位誤差のリスト
    # squared_error_lists = np.append(squared_error_lists, np.array([squared_error_list]), axis=0)

    # 測位可能確率の分布の更新とその平均の算出
    field_localizable_probability_distribution = localizable_probability_distribution.update(field_localizable_probability_distribution, grid_interval, targets, targets_localized)
    localizable_probability_avg = np.sum(field_localizable_probability_distribution[:, 2]*field_localizable_probability_distribution[:, 3])/np.sum(field_localizable_probability_distribution[:, 3])

    print("\r" + "{:.3f}".format((sim_cycle + 1)/sim_cycles*100) + "%" + " done." + " Average RMSE = " + "{:.4f}".format(root_mean_squared_error_avg) + " Average Localizable Probability = " + "{:.4f}".format(localizable_probability_avg), end="")
  print("\n")
  
  print(f"Average RMSE = {root_mean_squared_error_avg} m")

  # RMSEの累積分布関数を出力
  root_mean_squared_error_list_sorted = np.sort(root_mean_squared_error_list)
  cumulative_distribution_function = np.cumsum(root_mean_squared_error_list_sorted)/np.sum(root_mean_squared_error_list_sorted)
  cumulative_distribution_function_data = pd.DataFrame({
    "RMSE": root_mean_squared_error_list_sorted,
    "CDF": cumulative_distribution_function
  })
  cdf_filename = "cumulative_distribution_function.csv"
  cdf_filepath = os.path.join(output_dirpath, cdf_filename)
  cumulative_distribution_function_data.to_csv(cdf_filepath, index=False)
  print(f"{cdf_filename} was saved in {cdf_filepath}.")
  
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
  field_localizable_probability_distribution_data = pd.DataFrame({
    "x": field_localizable_probability_distribution[:, 0],
    "y": field_localizable_probability_distribution[:, 1],
    "localizable_probability": field_localizable_probability_distribution[:, 2],
    "data_count": field_localizable_probability_distribution[:, 3],
  })
  field_localizable_probability_distribution_filename = "field_localizable_probability_distribution.csv"
  field_localizable_probability_distribution_filepath = os.path.join(output_dirpath, field_localizable_probability_distribution_filename)
  field_localizable_probability_distribution_data.to_csv(field_localizable_probability_distribution_filepath, index=False)
  print(f"{field_localizable_probability_distribution_filename} was saved in {field_localizable_probability_distribution_filepath}.")

  # 測位順ごとにRMSEを算出
  # targets_order = np.arange(1, targets_count + 1)
  # order_to_root_mean_squared_error = np.sqrt(np.nanmean(squared_error_lists, axis=0))
  # order_to_root_mean_squared_error_data = pd.DataFrame({
  #   "localization_order": targets_order,
  #   "RMSE": order_to_root_mean_squared_error
  # })
  # order_to_root_mean_squared_error_filename = "order_to_root_mean_squared_error.csv"
  # order_to_root_mean_squared_error_filepath = os.path.join(output_dirpath, order_to_root_mean_squared_error_filename)
  # order_to_root_mean_squared_error_data.to_csv(order_to_root_mean_squared_error_filepath, index=False)
  # print(f"{order_to_root_mean_squared_error_filename} was saved in {order_to_root_mean_squared_error_filepath}.")


# メモ
# ホップした数の平均をとってそのRMSEを算出
# もう少し関数化できる
# collect_sample_dataやbuild_modelなどを関数化して同じconfigで一度に同時実行することも考えたが，サンプルデータやモデルのデータの容量などを考えると現行で問題ないと判断