import os
import sys
import numpy as np
import random
import yaml
import joblib
import pandas as pd
from datetime import datetime

# ランダムシードの設定
# random.seed(42)
# np.random.seed(42)

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

  # Feature 
  features_list = np.empty((0, 5))

  # Sample 
  sample_data_count = config["sample_data"]["count"]
  sample_data_filename = config["sample_data"]["filename"]
  sample_data_filepath = "../sample_data/" + sample_data_filename
  print(f"{sample_data_filename} will be saved in {sample_data_filepath}")

  # Model Config
  error_threshold = config["model"]["error_threshold"]

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
  # squared_error_lists = np.empty((0,20))

  # Temporary Parameter
  squared_error_total = 0.0 # シミュレーション全体における合計平方根誤差
  targets_localized_count_total = 0 # シミュレーション全体における合計ターゲット測位回数
  root_mean_squared_error_list = np.array([]) # シミュレーション全体におけるRMSEのリスト
  sim_cycle = 0
  
  print("\n")

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

        mask_sorted = np.argsort(np.linalg.norm(targets_estimated_initial - centroid_of_anchors, axis=1))[:1]
        # mask_sorted = np.argsort(np.linalg.norm(targets_estimated_initial - centroid_of_anchors, axis=1))

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

                in_anchors = np.array([any(np.all(np.isclose(sensor_available_for_target_estimated, anchors), axis=1)) for sensor_available_for_target_estimated in sensors_available_for_target_estimated])
                
                # is_recursion_available = np.any(~in_anchors) and recursive_count < recursive_count_max
                is_recursion_available = len(distances_estimated_for_target_estimated) > 3 and len(distances_estimated_for_target_estimated[~in_anchors]) > 0
                if not is_recursion_available:
                  break
                
                # RNのインデックスを取得
                mask_references = np.where(~in_anchors)[0]
                distances_estimated_for_target_estimated_from_references = distances_estimated_for_target_estimated[mask_references]
                index_distances_estimated_for_target_estimated_from_references_max = mask_references[np.argmax(distances_estimated_for_target_estimated_from_references)]

                distances_estimated_for_target_estimated = np.delete(distances_estimated_for_target_estimated, index_distances_estimated_for_target_estimated_from_references_max)
                sensors_available_for_target_estimated = np.delete(sensors_available_for_target_estimated, index_distances_estimated_for_target_estimated_from_references_max, axis=0)
                
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

                # 誤差が小さいとされる場合は終了
                is_positive = model.predict([features_recursively])
                if not is_positive:
                  target_estimated = target_estimated_recursively
                  break
                
                # 前回との測位誤差がerror_thresholdより大きい場合は終了
                if np.linalg.norm(target_estimated_recursively - target_estimated) < error_threshold:
                  target_estimated = target_estimated_recursively
                else:
                  break

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
  is_cooperative = config["localization"]["is_cooperative"]
  print("Localization: Least Square (LS) Method", end=" ")
  print("with Cooperation" if is_cooperative else "without Cooperation")

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

  targets = np.array([[field_range["x_top"]*i/8.0, field_range["y_top"]*j/8.0, 0.0] for i in [3, 5, 7] for j in [3, 5, 7]])
  targets_count: int = len(targets)
  print("target: (x, y) = ", end="")
  for target in targets:
    target_x = target[0]
    target_y = target[1]
    print(f"({target_x}, {target_y})", end=" ")
  print(f"\n=> anchor count: {len(targets)}")

  # Localization Config
  max_localization_loop = config["localization"]["max_loop"] # 最大測位回数
  channel = config["channel"] # LOSなどのチャネルを定義
  max_distance_measurement: int = config["localization"]["max_distance_measurement"] # 測距回数の最大（この回数が多いほど通信における再送回数が多くなる）
  newton_raphson_max_loop: int = config["localization"]["newton_raphson"]["max_loop"] # Newton Raphson 計算回数の最大
  newton_raphson_threshold: float = eval(config["localization"]["newton_raphson"]["threshold"]) # Newton Raphson 閾値

  # Feature 
  features_list = np.empty((0, 5))

  # Sample
  sample_data_count = config["sample_data"]["count"]
  sample_data_filename = config["sample_data"]["filename"]
  sample_data_filepath = "../sample_data/" + sample_data_filename
  print(f"{sample_data_filename} will be saved in {sample_data_filepath}")

  # Model
  error_threshold = config["model"]["error_threshold"]
  
  # Temporary Parameter
  squared_error_total = 0.0 # シミュレーション全体における合計平方根誤差
  targets_localized_count_total = 0 # シミュレーション全体における合計ターゲット測位回数
  root_mean_squared_error_list = np.array([]) # シミュレーション全体におけるRMSEのリスト
  sim_cycle = 0
  
  print("\n")

  # シミュレーション開始
  while np.sum(features_list[:, -1] < error_threshold) < sample_data_count or np.sum(features_list[:, -1] >= error_threshold) < sample_data_count:
    # sensor は anchor node と reference node で構成
    anchors = np.array([[anchor_config["x"], anchor_config["y"], 1] for anchor_config in anchors_config])
    sensors_original: np.ndarray = np.copy(anchors) # 実際の座標
    sensors: np.ndarray = np.copy(sensors_original) # anchor以外は推定座標

    # ターゲット
    # targets: np.ndarray = np.array([[round(random.uniform(0.0, width), 2), round(random.uniform(0.0, height), 2), 0] for target_count in range(targets_count)])
    # 測位フラッグのリセット
    targets[:, 2] = 0.0
    rng = np.random.default_rng()
    rng.shuffle(targets, axis=0)

    # 平方根誤差のリスト
    squared_error_list = np.array([])

    for localization_loop in range(max_localization_loop): # unavailableの補完 本来はWhileですべてのTNが"is_localized": 1 になるようにするのがよいが計算時間短縮のため10回に設定してある（とはいってもほとんど測位されてました）
      for target in targets:
        distances_measured: np.ndarray = np.array([])
        rss_list = np.zeros(len(anchors))
        if target[2] == 0: # i番目のTNがまだ測位されていなければ行う
          for sensor_original, sensor in zip(sensors_original, sensors):
            distance_accurate = np.linalg.norm(target[:2] - sensor_original[:2])
            distance_measured, rx_power = distance_toa.calculate(channel, max_distance_measurement, distance_accurate)
            distances_measured = np.append(distances_measured, distance_measured)

        # 三辺測量の条件（LOPの初期解を導出できる条件）
        distances_estimated = distances_measured[~np.isinf(distances_measured)]
        is_localizable = len(distances_estimated) >= 3
        if not is_localizable:
          continue

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
          feature_distance_from_sensors_to_approximate_line = distance_from_sensors_to_approximate_line.calculate(sensors_available)
          feature_residual_avg = residual_avg.calculate(sensors_available, distances_estimated, target_estimated)

          features = np.array([
            feature_convex_hull_volume,
            feature_distance_from_center_of_field_to_target,
            feature_distance_from_sensors_to_approximate_line,
            feature_residual_avg,
          ])

          # 平均平方根誤差の算出
          squared_error = distance_error_squared.calculate(target, target_estimated)
          squared_error_list = np.append(squared_error_list, squared_error)
          
          # 誤差の特徴量はfeaturesの配列の一番最後に
          feature_error = np.sqrt(squared_error)
          features = np.append(features, feature_error)
          if feature_error < error_threshold and np.sum(features_list[:, -1] < error_threshold) < sample_data_count:
            features_list = np.append(features_list, [features], axis=0)
          if feature_error >= error_threshold and np.sum(features_list[:, -1] >= error_threshold) < sample_data_count:
            features_list = np.append(features_list, [features], axis=0)

          # 測位フラグの更新
          target[2], target_estimated[2] = 1, 1
          targets_localized = targets[targets[:, 2] == 1]
          if len(targets_localized) == targets_count:
            break

          if is_cooperative:
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
    print(f"positive: {positive}/{sample_data_count} negative: {negative}/{sample_data_count}", end=" ")
    print("Average RMSE = " + "{:.4f}".format(root_mean_squared_error_avg) + " Average Localizable Probability = " + "{:.4f}".format(localizable_probability_avg), end="\r\r")
  print("\n")

  print(f"Average RMSE = {root_mean_squared_error_avg} m")

  features_data = pd.DataFrame({
    "convex_hull_volume": features_list[:, 0], 
    "distance_from_center_of_field_to_target": features_list[:, 1],
    "distance_from_sensors_to_approximate_line": features_list[:, 2],
    "residual_avg": features_list[:, 3],
    "error": features_list[:, 4]
  })

  features_data.to_csv(sample_data_filepath, index=False)
  print(f"{sample_data_filename} was saved in {sample_data_filepath}")

  print("\ncomplete.")