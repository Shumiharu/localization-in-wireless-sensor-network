import os
import sys
import time
import numpy as np
import random
import yaml
import joblib
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ランダムシードの設定
# random.seed(42)
# np.random.seed(42)

# 基本関数
from basis import distance_toa
from basis import normalization
from basis import line_of_position
from basis import newton_raphson
from basis import target_coordinates

# 特徴量の算出
from feature import distance_from_sensors_to_approximate_line
from feature import distance_from_center_of_field_to_target
from feature import convex_hull_volume
from feature import residual_avg
from feature import distance_error_squared
from feature import feature_extraction

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
    print(f"{config_filename} was loaded from {config_filepath}\n")

  # Localization Config
  is_successive: bool = config["localization"]["is_successive"]
  is_cooperative: bool = config["localization"]["is_cooperative"]
  is_predictive: bool = config["localization"]["is_predictive"]
  is_recursive: bool = config["localization"]["is_recursive"]

  max_localization_loop: int = config["localization"]["max_loop"] # 最大測位回数
  max_distance_measurement: int = config["localization"]["max_distance_measurement"] # 最大測距回数（この回数が多いほど通信における再送回数が多くなる）

  newton_raphson_max_loop: int = config["localization"]["newton_raphson"]["max_loop"] # Newton Raphson 計算回数の最大
  newton_raphson_threshold: float = eval(config["localization"]["newton_raphson"]["threshold"]) # Newton Raphson 閾値

  print("Localization: Least Square (LS) Method", end=" ")
  print("with Cooperation" if is_cooperative else "without Cooperation", end=" ")
  print("Collectively" if not is_successive else "Successively (Conventional Algorithm)")

  # Learning Model
  if is_predictive:
    error_threshold = config["model"]["error_threshold"]
    model_filename = config["model"]["filename"]
    model_filepath = "../models/" + model_filename
    model = joblib.load(model_filepath)
    print("Error 'Recursive' Prediction" if is_recursive else "Error Prediction", end=" ")
    print(f"by Machine Learning (model: {model_filename})\n")
  else:
    print("No Error Prediction\n")

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

  # Channel Config
  channel = config["channel"] # LOSなどのチャネルを定義

  # Anchors & Targets Config
  anchors_config = config["anchors"]
  anchors = np.array([[anchor_config["x"], anchor_config["y"], 0] for anchor_config in anchors_config])
  centroid_of_anchors = np.array([np.mean(anchors[:, 0]), np.mean(anchors[:, 1])])
  print("anchor: (x, y) = ", end="")
  for anchor_config in anchors_config:
    anchor_x = anchor_config["x"]
    anchor_y = anchor_config["y"]
    print(f"({anchor_x}, {anchor_y})", end=" ")
  print(f"\n=> anchor count: {len(anchors_config)}")

  targets_count: int = config["targets"]["count"]
  print("target: (x, y) = random")
  print(f"=> target count: {targets_count}\n")

  # Fingerprint Config
  # fingerprint_filename = "fingerprint_0.csv"
  # fingerprint_filepath = "../fingerprint/" + fingerprint_filename
  # if is_subprocess:
  #   fingerprint_filepath = os.path.join(args[1], "fingerprint.csv")
  # fingerprint_data = pd.read_csv(fingerprint_filepath)
  # fingerprint_list = fingerprint_data.to_numpy()
  # print(f"{fingerprint_filename} was loaded from {fingerprint_filepath}")

  # Simulation Cycle
  sim_cycles = config["sim_cycles"] # シミュレーション回数
  print(f"Simulation Cycle: {sim_cycles}\n")

  # Temporary Parameter
  squared_error_total = 0.0 # シミュレーション全体における合計平方根誤差
  targets_localized_count_total = 0 # シミュレーション全体における合計ターゲット測位回数
  root_mean_squared_error_list = np.array([]) # シミュレーション全体におけるRMSEのリスト
  is_successive = True
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
    sensors_original = np.copy(anchors) # 実際の座標
    sensors = np.copy(sensors_original) # anchor以外は推定座標

    # ターゲット
    targets: np.ndarray = np.array([[round(random.uniform(0.0, width), 2), round(random.uniform(0.0, height), 2), 0] for target_count in range(targets_count)])
    np.random.shuffle(targets)

    # 平方根誤差のリスト
    squared_error_list = np.array([])
    # squared_error_list = np.array([np.nan]*targets_count)

    distances_measured_list = np.empty((0, len(targets)))
    targets_localized = np.empty((0, 3))
    targets_unlocalized_count = np.zeros(len(targets))
    index_targets_end = 0

    # 測位開始時間を取得
    time_localization_start = time.time()

    is_localizable = True
    while is_localizable:

      # 測距値のリセット
      if is_successive:
        distances_measured_list = np.empty((0, len(targets)))

      # 測距フェーズ
      mask_targets_unlocalized_original = np.where(targets[:, 2] == 0)[0]
      shift_targets_begin = np.argmax(mask_targets_unlocalized_original >= index_targets_end)
      mask_targets_unlocalized = np.roll(mask_targets_unlocalized_original, -shift_targets_begin)
      mask_sensors_unmeasured = np.where(sensors[:, 2] == 0)[0]

      # print(f"mask_targets_unlocalized_original: {np.where(targets[:, 2] == 0)[0]}")
      # print(f"index_targets_end: {index_targets_end}")
      # print(f"shift_targets_begin: {shift_targets_begin}")
      # print(f"mask_targets_unlocalized: {mask_targets_unlocalized}")

      distances_measured_list = np.array([
        [
          distance_toa.calculate(channel, max_distance_measurement, np.linalg.norm(target[:2] - sensor_original[:2]))[0] if index_target in mask_targets_unlocalized else np.nan
          for index_target, target in enumerate(targets)
        ]
        for sensor_original in sensors_original[mask_sensors_unmeasured]
      ])
        
      # 測距フラグの更新
      if not is_successive:
        sensors[mask_sensors_unmeasured, 2] = 1

      # 一時測位フェーズ
      targets_estimated_initial = np.empty((0, 2))
      mask_targets_estimated_initial = np.array([], dtype="int")
      distances_measured_list_transposed = distances_measured_list.T
      for index_targets_unlocalized, distances_measured_for_targets_unlocalized in zip(mask_targets_unlocalized, distances_measured_list_transposed[mask_targets_unlocalized]):
        if targets_unlocalized_count[index_targets_unlocalized] < max_localization_loop:

          mask_distance_measurable_for_targets_unlocalized = ~np.isinf(distances_measured_for_targets_unlocalized)
          distances_estimated_for_targets_unlocalized = distances_measured_for_targets_unlocalized[mask_distance_measurable_for_targets_unlocalized]
          sensors_available_for_targets_unlocalized = sensors[mask_distance_measurable_for_targets_unlocalized]
          if len(distances_estimated_for_targets_unlocalized) >= 3:

            target_estimated_initial = target_coordinates.calculate(
              sensors_available_for_targets_unlocalized,
              distances_estimated_for_targets_unlocalized,
              newton_raphson_max_loop,
              newton_raphson_threshold,
              field_range
            )

            if not np.any(np.isnan(target_estimated_initial)):
              targets_estimated_initial = np.append(targets_estimated_initial, [target_estimated_initial], axis=0)
              mask_targets_estimated_initial = np.append(mask_targets_estimated_initial, index_targets_unlocalized)
              if is_successive:
                break

          else:
            targets_unlocalized_count[index_targets_unlocalized] += 1
      
      # 座標決定フェーズ
      if len(targets_estimated_initial) > 0:

        # mask_sorted = np.argsort(np.linalg.norm(targets_estimated_initial - centroid_of_anchors, axis=1))[:1]
        # mask_sorted = np.argsort(np.linalg.norm(targets_estimated_initial - centroid_of_anchors, axis=1))
        # mask_sorted = np.array([np.random.choice(len(mask_targets_estimated_initial))])
        if is_successive:
          mask_sorted = np.array([0])

        targets_estimated = targets_estimated_initial[mask_sorted]
        mask_targets_estimated = mask_targets_estimated_initial[mask_sorted]
        for index_targets_estimated, target_estimated in zip(mask_targets_estimated, targets_estimated):
          
          # 実際の座標を取得
          target = targets[index_targets_estimated]
          # print(f"target: {target}")

          distances_measured_for_target_estimated = distances_measured_list_transposed[index_targets_estimated]
          mask_distance_measurable_for_target_estimated = ~np.isinf(distances_measured_for_target_estimated)

          distances_estimated_for_target_estimated = distances_measured_for_target_estimated[mask_distance_measurable_for_target_estimated]
          sensors_available_for_target_estimated = sensors[:len(mask_distance_measurable_for_target_estimated)][mask_distance_measurable_for_target_estimated]

          if is_predictive:
            
            # 特徴量の算出
            features = feature_extraction.calculate(
              sensors_available_for_target_estimated,
              distances_estimated_for_target_estimated,
              target_estimated,
              field_range
            )

            # 陽性判定
            is_positive = model.predict([features])
            if is_positive and is_recursive:

              recursive_count = 0
              while len(distances_estimated_for_target_estimated) >= 3:

                in_anchors = np.array([any(np.all(sensor_available_for_target_estimated == anchors, axis=1)) for sensor_available_for_target_estimated in sensors_available_for_target_estimated])
                if recursive_count == 0:
                  recursive_count_max = len(sensors_available_for_target_estimated[~in_anchors])
                  # print(f"\nrecursive count max: {recursive_count_max}")
                
                is_recursion_available = np.any(~in_anchors) and recursive_count < recursive_count_max
                if not is_recursion_available:
                  break

                mask_references = np.where(~in_anchors)[0]
                distances_estimated_for_target_estimated_from_references = distances_estimated_for_target_estimated[mask_references]
                index_distances_estimated_for_target_estimated_from_references_max = mask_references[np.argmax(distances_estimated_for_target_estimated_from_references)]

                distances_estimated_for_target_estimated = np.delete(distances_estimated_for_target_estimated, index_distances_estimated_for_target_estimated_from_references_max)
                sensors_available_for_target_estimated = np.delete(sensors_available_for_target_estimated, index_distances_estimated_for_target_estimated_from_references_max, axis=0)
                
                if len(distances_estimated_for_target_estimated) >= 3:

                  target_estimated_recursively = target_coordinates.calculate(
                    sensors_available_for_target_estimated,
                    distances_estimated_for_target_estimated,
                    newton_raphson_max_loop,
                    newton_raphson_threshold,
                    field_range
                  )

                  features_recursively = feature_extraction.calculate(
                    sensors_available_for_target_estimated,
                    distances_estimated_for_target_estimated,
                    target_estimated_recursively,
                    field_range
                  )

                  is_positive = model.predict([features_recursively])
                  if not is_positive:
                    target_estimated = target_estimated_recursively
                    break

                  if np.linalg.norm(target_estimated_recursively - target_estimated) < error_threshold:
                    target_estimated = target_estimated_recursively
                  else:
                    break

                recursive_count += 1

          if not is_predictive or not is_positive:

            # 推定座標の確定
            target_localized = np.append(target_estimated, 0)
            targets_localized = np.append(targets_localized, [target_localized], axis=0)
            # print(f"target_localized[{index_targets_estimated_sorted}]: {target_localized}\n")

            # 二乗誤差の算出
            squared_error = distance_error_squared.calculate(target, target_localized)
            squared_error_list = np.append(squared_error_list, squared_error)

            # 協調測位であれば測位したTNをSNに追加する（RNに変更する）
            if is_cooperative:
              sensors_original = np.append(sensors_original, [target], axis=0)
              sensors = np.append(sensors, [target_localized], axis=0)

            # 測位フラグの更新
            targets[index_targets_estimated, 2] = 1
        
        if is_successive:
          index_targets_end = np.max(mask_targets_estimated)

      else:
        is_localizable = np.any(targets_unlocalized_count[mask_targets_unlocalized] != max_localization_loop)

      # if len(targets[targets[:, 2] == 1]) == targets_count:
      if len(targets_localized) == targets_count:
        break
    
    # 測位時間の算出
    time_localization_end = time.time()
    duration_localization_per_target = (time_localization_end - time_localization_start)/len(targets_localized)
    if sim_cycle == 0:
      duration_localization_per_target_avg = duration_localization_per_target
    else:
      duration_localization_per_target_avg = (duration_localization_per_target_avg*sim_cycle + duration_localization_per_target)/(sim_cycle + 1)

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

    print("\r" + "{:.3f}".format((sim_cycle + 1)/sim_cycles*100) + "%" + " done." + " / RMSE: " + "{:.4f}".format(root_mean_squared_error_avg) + " / Avg. Localizable Prob.: " + "{:.4f}".format(localizable_probability_avg) + " / Avg. Localization Duration per target: " + "{:.4f}".format(duration_localization_per_target_avg), end="")
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
# collect_sample_dataやbuild_modelなどを関数化して同じconfigで一度に同時実行することも考えたが，サンプルデータやモデルのデータの容量などを考えると現行で問題ないと判断

# anchor nodeによる測距
# targets_estimated = np.empty((0,2))
# distances_measured_list = np.empty((0, len(anchors)))
# for target in targets:
#   distances_measured = np.array([])
#   rx_power_list = np.array([])
#   for anchor in anchors:
#     distance_accurate = np.linalg.norm(target[:2] - anchor[:2])
#     distance_measured, rx_power = distance_toa.calculate(channel, max_distance_measurement, distance_accurate)
#     distances_measured = np.append(distances_measured, distance_measured)
#     rx_power_list = np.append(rx_power_list, rx_power)
#   print(fingerprint_list[:, 2:])
#   print(rx_power_list)
#   print(np.square(fingerprint_list[:, 2:] - rx_power_list))
#   target_estimated = fingerprint_list[np.nanargmin(np.nansum(np.square(fingerprint_list[:, 2:] - rx_power_list), axis=1)), :2]
#   targets_estimated = np.append(targets_estimated, [target_estimated], axis=0)
#   distances_measured_list = np.append(distances_measured_list, [distances_measured], axis=0)
# print(targets)
# print(targets_estimated)
# indices = np.argsort(np.linalg.norm(targets_estimated - centroid_of_anchors, axis=1))
# targets = targets[indices]
# distances_measured_list = distances_measured_list[indices]
# for target in targets:
#   a = np.linalg.norm(target[:2] - centroid_of_anchors)
#   print(f"distance = {a}")



    #   is_initial_judge = True
    #   target_estimated_mean = np.zeros(len(target))
    #   recursive_count = 0

    #   while len(distances_estimated) >= 3:

    #     # 測位
    #     target_estimated = line_of_position.calculate(sensors_available, distances_estimated) # Line of Positionによる初期解の算出
    #     target_estimated = newton_raphson.calculate(sensors_available, distances_estimated, target_estimated, newton_raphson_max_loop, newton_raphson_threshold) # Newton Raphson法による最適解の算出
    #     target_estimated = normalization.calculate(field_range, target_estimated) # 測位フィールド外に測位した場合の補正
    #     target_estimated = np.append(target_estimated, 0) # 測位フラグの付加
        
    #     if not np.any(np.isnan(target_estimated)):
          
    #       if is_predictive:

    #         # 特徴量の計算
            # feature_convex_hull_volume = convex_hull_volume.calculate(sensors_available)
            # feature_distance_from_center_of_field_to_target = distance_from_center_of_field_to_target.calculate(field_range, target_estimated)
            # feature_distance_from_sensors_to_approximate_line = distance_from_sensors_to_approximate_line.calculate(sensors_available)
            # feature_residual_avg = residual_avg.calculate(sensors_available, distances_estimated, target_estimated)

            # features = np.array([
            #   feature_convex_hull_volume,
            #   feature_distance_from_center_of_field_to_target,
            #   feature_distance_from_sensors_to_approximate_line,
            #   feature_residual_avg,
            # ]) 

    #       if not is_predictive or not model.predict([features]):
            
    #         # 平均平方根誤差の算出
    #         squared_error = distance_error_squared.calculate(target, target_estimated)
    #         squared_error_list = np.append(squared_error_list, squared_error)
    #         # order_localized = len(targets_localized) - 1
    #         # squared_error_list[order_localized] = squared_error

    #         # 測位フラグの更新
    #         target[2], target_estimated[2] = 1, 1

    #         # 協調測位の場合はReference Nodeとしてセンサを追加する
    #         if is_cooperative:
    #           sensors_original = np.append(sensors_original, [target], axis=0)
    #           sensors = np.append(sensors, [target_estimated], axis=0)

    #         break

    #       else:
    #         if is_recursive:
    #           if is_initial_judge or np.linalg.norm(target_estimated[:2] - target_estimated_previous[:2]) < error_threshold:
    #             in_anchors = np.array([any(np.all(sensor_available == anchors, axis=1)) for sensor_available in sensors_available])
                
    #             if np.all(in_anchors) and recursive_count == recursive_count_max: # ここのif文をis_initial_judgeにすると測位確率は99.99%になるが測位精度が大きく劣化

    #               # 平均平方根誤差の算出
    #               squared_error = distance_error_squared.calculate(target, target_estimated_mean)
    #               squared_error_list = np.append(squared_error_list, squared_error)

    #               # 測位フラグの更新
    #               target[2], target_estimated_mean[2] = 1, 1

    #               # 協調測位の場合はReference Nodeとしてセンサを追加する
    #               if is_cooperative:
    #                 sensors_original = np.append(sensors_original, [target], axis=0)
    #                 sensors = np.append(sensors, [target_estimated_mean], axis=0)

    #               break

    #             if not np.all(in_anchors):
    #               if is_initial_judge:
    #                 is_initial_judge = False
    #                 recursive_count_max = len(sensors_available[~in_anchors])

    #               mask_rn = np.where(~in_anchors)[0]
    #               distances_estimated_from_rn = distances_estimated[mask_rn]
    #               mask_rn_max = mask_rn[np.argmax(distances_estimated_from_rn)]

    #               distances_estimated = np.delete(distances_estimated, mask_rn_max)
    #               sensors_available = np.delete(sensors_available, mask_rn_max, axis=0)

    #               target_estimated_previous = target_estimated
    #               target_estimated_mean = (target_estimated_mean*recursive_count + target_estimated)/(recursive_count + 1)

    #               recursive_count += 1
    #             else:
    #               break
    #           else:
    #             break
    #         else:
    #           break
    #     else:
    #       break

    #   targets_localized = targets[targets[:, 2] == 1] # 推定座標ではないので注意
    #   if len(targets_localized) == targets_count:
    #     break
    # else:
    #   continue
  # break

      # targets = np.array([
    #   [29.07, 6.51,  0.],
    #   [26.  ,  1.51,  0.],
    #   [24.27,  9.13,  1.  ],
    #   [11.56,  9.25,  1.  ],
    #   [22.2 ,  9.23,  1.  ],
    #   [11.52,  9.6 ,  1.  ],
    #   [15.9 , 11.12,  1.  ],
    #   [ 2.94, 20.45,  1.  ],
    #   [27.61, 23.74,  1.  ],
    #   [29.58, 28.73,  0.  ],
    #   [28.54, 16.64,  1.  ],
    #   [26.18, 14.76,  1.  ],
    #   [22.28,  5.84,  1.  ],
    #   [4.95, 3.95, 1.  ],
    #   [ 8.08, 27.02,  1.  ],
    #   [15.63, 20.38,  1.  ],
    #   [24.59, 28.69,  1.  ],
    #   [ 2.4 , 22.39,  0.  ],
    #   [15.75, 25.06,  1.  ],
    #   [9.76, 7.18, 1.  ]
    # ])
    # targets = np.array([
    #   [29.07, 6.51,  0.],
    #   [26.  ,  1.51,  0.],
    #   [24.27,  9.13,  0.  ],
    #   [11.56,  9.25,  0.  ],
    #   [22.2 ,  9.23,  0.  ],
    #   [11.52,  9.6 ,  0.  ],
    #   [15.9 , 11.12,  0.  ],
    #   [ 2.94, 20.45,  0.  ],
    #   [27.61, 23.74,  0.  ],
    #   [29.58, 28.73,  0.  ],
    #   [28.54, 16.64,  0.  ],
    #   [26.18, 14.76,  0.  ],
    #   [22.28,  5.84,  0.  ],
    #   [4.95, 3.95, 0.  ],
    #   [ 8.08, 27.02,  0.  ],
    #   [15.63, 20.38,  0.  ],
    #   [24.59, 28.69,  0.  ],
    #   [ 2.4 , 22.39,  0.  ],
    #   [15.75, 25.06,  0.  ],
    #   [9.76, 7.18, 0.  ]
    # ])
    
       # targets = np.array([
    #   [29.07, 6.51,  0.],
    #   [26.  ,  1.51,  0.],
    #   [24.27,  9.13,  0.  ],
    #   [11.56,  9.25,  0.  ],
    #   [22.2 ,  9.23,  0.  ],
    #   [11.52,  9.6 ,  0.  ],
    #   [15.9 , 11.12,  0.  ],
    #   [ 2.94, 20.45,  0.  ],
    #   [27.61, 23.74,  0.  ],
    #   [29.58, 28.73,  0.  ],
    #   [28.54, 16.64,  0.  ],
    #   [26.18, 14.76,  0.  ],
    #   [22.28,  5.84,  0.  ],
    #   [4.95, 3.95, 0.  ],
    #   [ 8.08, 27.02,  0.  ],
    #   [15.63, 20.38,  0.  ],
    #   [24.59, 28.69,  0.  ],
    #   [ 2.4 , 22.39,  0.  ],
    #   [15.75, 25.06,  0.  ],
    #   [9.76, 7.18, 0.  ]
    # ])


        #   for index_sensor_unmeasured in mask_sensors_unmeasured:
        # distances_measured = np.full(len(targets), np.nan)
        # for index_targets_unlocalized in mask_targets_unlocalized:
        #   distance_accurate = np.linalg.norm(targets[index_targets_unlocalized, :2] - sensors_original[index_sensor_unmeasured, :2])
        #   distance_measured, rx_power = distance_toa.calculate(channel, max_distance_measurement, distance_accurate)
        #   distances_measured[index_targets_unlocalized] = distance_measured
        # distances_measured_list = np.append(distances_measured_list, [distances_measured], axis=0)