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
import time

# ランダムシードの設定
# random.seed(42)
# np.random.seed(42)

# 基本関数
from basis import distance_toa
from basis import target_coordinates
from basis import distances_avg

# 特徴量の算出
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
    print(f"\n{config_filename} was loaded from {config_filepath}\n")

  # Localization Config
  is_successive: bool = config["localization"]["is_successive"]
  is_cooperative: bool = config["localization"]["is_cooperative"]
  is_predictive: bool = config["localization"]["is_predictive"]
  is_recursive: bool = config["localization"]["is_recursive"]
  is_sorted = config["localization"]["is_sorted"]
  # is_division: bool = config["localization"]["is_division"]

  max_localization_loop: int = config["localization"]["max_loop"] # 最大測位回数
  max_distance_measurement: int = config["localization"]["max_distance_measurement"] # 最大測距回数（この回数が多いほど通信における再送回数が多くなる）

  newton_raphson_max_loop: int = config["localization"]["newton_raphson"]["max_loop"] # Newton Raphson 計算回数の最大
  newton_raphson_threshold: float = eval(config["localization"]["newton_raphson"]["threshold"]) # Newton Raphson 閾値

  print("Localization: Least Square (LS) Method", end=" ")
  print("with Cooperation" if is_cooperative else "without Cooperation", end=" ")
  print("'Collectively'" if not is_successive else "'Successively (Conventional Algorithm)'")

  print("\nEstimated targets (variable: targets_estimated) are localized", end=" ")
  print("in order from the center." if is_sorted else "in that order.")

  # Learning Model
  if is_predictive:
    model_type = config["model"]["type"]
    is_built_successively = config["model"]["is_built_successively"]
    model_filename = config["model"]["filename"]
    model_subdirname = "successive" if is_built_successively else "collective" # 測位がcollectiveでもモデルはsuccessiveを選択できる
    model_filepath = f"../models/{model_subdirname}/{model_type}/{model_filename}"
    model = joblib.load(model_filepath)    
    print("\nError 'Recursive' Prediction" if is_recursive else "\nError Prediction", end=" ")
    print(f"by Machine Learning (model: {model_type} -> filepath: {model_filepath})\n")

    error_threshold = config["model"]["error_threshold"]
  else:
    print("\nNo Error Prediction\n")

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

  localize_time_ave_total = 0.0    
  localize_time_list = np.array([])
  sim_time_start = time.time()

  big_error_count = 0

  # シミュレーション開始
  for sim_cycle in range(sim_cycles):

    # sensor は anchor と reference で構成
    sensors_original = np.copy(anchors) # 実際の座標
    sensors = np.copy(sensors_original) # anchor以外は推定座標

    # ターゲット
    targets: np.ndarray = np.array([[round(random.uniform(0.0, width), 2), round(random.uniform(0.0, height), 2), 0] for target_count in range(targets_count)])
    np.random.shuffle(targets)

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

    # 平方根誤差のリスト
    squared_error_list = np.array([])
    # squared_error_list = np.array([np.nan]*targets_count)

    # distances_measured_list = np.empty((0, len(targets)))
    targets_localized = np.empty((0, 3))
    targets_unlocalized_count = np.zeros(len(targets))
    # distances_measured_list_count = np.empty((0, len(targets)))
    index_targets_begin = 0
    is_initial_distances_measurement = True

    # 測位開始時間を取得
    time_localization_start = time.time()

    is_localizable = True
    while is_localizable:

      # 測距値のリセット
      # if is_successive:
      #   distances_measured_list = np.empty((0, len(targets)))

      # 測距フェーズ
      mask_targets_unlocalized_original = np.where(targets[:, 2] == 0)[0]
      shift_targets_begin = np.argmax(mask_targets_unlocalized_original >= index_targets_begin)
      mask_targets_unlocalized = np.roll(mask_targets_unlocalized_original, -shift_targets_begin)
      mask_sensors_unmeasured = np.where(sensors[:, 2] == 0)[0]

      # print(f"mask_targets_unlocalized_original: {np.where(targets[:, 2] == 0)[0]}")
      # print(f"index_targets_begin: {index_targets_begin}")
      # print(f"shift_targets_begin: {shift_targets_begin}")
      # print(f"mask_targets_unlocalized: {mask_targets_unlocalized}")

      # 測距値の算出
      distances_measured_list = np.array([
        [
          distance_toa.calculate(channel, max_distance_measurement, np.linalg.norm(target[:2] - sensor_original[:2]))[0] if index_target in mask_targets_unlocalized else np.nan
          for index_target, target in enumerate(targets)
        ]
        for sensor_original in sensors_original[mask_sensors_unmeasured]
      ])

      # 平均測距値の算出（今回の試行で測距不能でも，前回の試行で測距値が得られていたならばそちらを利用する）
      if is_initial_distances_measurement: 
        distances_measured_list_avg = distances_measured_list
        distances_measured_list_count = np.where(np.isinf(distances_measured_list), 0, 1)
        is_initial_distances_measurement = False
      else:
        distances_measured_list_avg, distances_measured_list_count = distances_avg.calculate(distances_measured_list_avg, distances_measured_list, distances_measured_list_count)
        
      # 測距フラグの更新
      # if not is_successive:
      #   sensors[mask_sensors_unmeasured, 2] = 1

      # 一時測位フェーズ
      targets_estimated_initial = np.empty((0, 2))
      mask_targets_estimated_initial = np.array([], dtype="int")
      # distances_measured_list_transposed = distances_measured_list.T
      distances_measured_list_avg_transposed = distances_measured_list_avg.T
      
      for index_targets_unlocalized, distances_measured_avg_for_targets_unlocalized in zip(mask_targets_unlocalized, distances_measured_list_avg_transposed[mask_targets_unlocalized]):
        
        # 最大測位回数を超えてなければ推定座標を算出
        if targets_unlocalized_count[index_targets_unlocalized] < max_localization_loop:

          mask_distance_measurable_for_targets_unlocalized = ~np.isinf(distances_measured_avg_for_targets_unlocalized)
          distances_estimated_for_targets_unlocalized = distances_measured_avg_for_targets_unlocalized[mask_distance_measurable_for_targets_unlocalized]
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
              if is_successive and not is_sorted:
                break
            # else:
            #   targets_unlocalized_count[index_targets_unlocalized] += 1

          else:
            # 一時測位できない場合は加算
            targets_unlocalized_count[index_targets_unlocalized] += 1
      
      # 座標決定フェーズ
      if len(targets_estimated_initial) > 0:
        
        # 測位した順番に座標を決定（デフォルト）
        mask_sorted = np.arange(len(targets_estimated_initial))

        if is_sorted:
          # ANの中心に近い方から座標を順に決定
          mask_sorted = np.argsort(np.linalg.norm(targets_estimated_initial - centroid_of_anchors, axis=1))
          if is_successive:
            mask_sorted = mask_sorted[:1]

        targets_estimated = targets_estimated_initial[mask_sorted]
        mask_targets_estimated = mask_targets_estimated_initial[mask_sorted]
        for index_targets_estimated, target_estimated in zip(mask_targets_estimated, targets_estimated):
          
          # 実際の座標を取得
          target = targets[index_targets_estimated]
          # print(f"target: {target}")

          distances_measured_avg_for_target_estimated = distances_measured_list_avg_transposed[index_targets_estimated]
          mask_distance_measurable_for_target_estimated = ~np.isinf(distances_measured_avg_for_target_estimated)

          distances_estimated_for_target_estimated = distances_measured_avg_for_target_estimated[mask_distance_measurable_for_target_estimated]
          sensors_available_for_target_estimated = sensors[:len(mask_distance_measurable_for_target_estimated)][mask_distance_measurable_for_target_estimated]
          sensors_available_for_target_estimated_orignal = np.copy(sensors_available_for_target_estimated)

          if is_predictive:
            
            # 特徴量の算出
            features = feature_extraction.calculate(
              sensors_available_for_target_estimated,
              distances_estimated_for_target_estimated,
              target_estimated,
              field_range
            )
            # print(f"features: {features}")

            # 陽性判定 (モデルでは0～5を出力し，それを最大値×2で割ることで確率として考えて，50%を最小として重みに則って確率的に判定しようとしている)
            is_positive = model.predict([features])

            # 閾値分割する場合  (閾値が2.0以上は従来手法どうりに必ず再帰にかける)
            is_positive_probable = is_positive/4
            if is_positive_probable == 0:
              is_positive = random.choices([0,1],weights=[is_positive_probable,(1.0-is_positive_probable)],k=1)
            
            else:
              is_positive = random.choices([0,1],weights=[is_positive_probable,(1.0-is_positive_probable)],k=1)
            # print(f"{is_positive}")
            # print(f"{is_recursive}")


            # if is_positive and is_recursive:

            if is_positive[0] == 1 and is_recursive:
              # if is_positive_probable == 0.4:
                # print("なぜ")

              recursive_count = 0
              targets_estimated_recursively = np.array([target_estimated])
              while len(distances_estimated_for_target_estimated) >= 3:

                in_anchors = np.array([any(np.all(np.isclose(sensor_available_for_target_estimated, anchors), axis=1)) for sensor_available_for_target_estimated in sensors_available_for_target_estimated])
                # RNのインデックスを取得
                mask_references = np.where(~in_anchors)[0]
                if recursive_count == 0:
                  recursive_count_max = np.sum(~in_anchors)

                # 再帰を行うために必要な距離情報とアンカーノード情報があるかどうか判定する．
                is_recursion_available = len(distances_estimated_for_target_estimated) > 3 and len(distances_estimated_for_target_estimated[~in_anchors]) > 0
                # 再帰条件を満たしていない場合
                if not is_recursion_available:
                  # 再帰回数が上限となる場合
                  if recursive_count == recursive_count_max > 0:
                    # 陰性データと判別し，再帰した測位データの平均を測位データとして考える．
                    
                    # is_positive = False
                    is_positive[0] = 1

                    target_estimated = np.mean(targets_estimated_recursively, axis=0)
                  break
                
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
                
                is_positive_probable = is_positive/4
                if is_positive_probable == 0:
                  is_positive = random.choices([0,1],weights=[is_positive_probable,(1.0-is_positive_probable)],k=1)            
                else:
                  is_positive = random.choices([0,1],weights=[is_positive_probable,(1.0-is_positive_probable)],k=1)

                # if not is_positive:
                # 陰性データと判断された場合
                if is_positive[0] == 0:
                  target_estimated = target_estimated_recursively
                  # print(f"{target_estimated_recursively}")
                  # print(f"{np.linalg.norm(target_estimated_recursively)}")
                  break
                
                # 前回との測位誤差がerror_thresholdより大きい場合は終了
                if np.linalg.norm(target_estimated_recursively - target_estimated) < error_threshold:
                  target_estimated = target_estimated_recursively
                  targets_estimated_recursively = np.append(targets_estimated_recursively, [target_estimated_recursively], axis=0)
                else:
                  break
                recursive_count += 1

          # if not is_positive or not is_predictive:
          if is_positive[0] == 0 or not is_predictive:

            # 推定座標の確定
            target_localized = np.append(target_estimated, 0)
            targets_localized = np.append(targets_localized, [target_localized], axis=0)
            # print(f"target_localized[{index_targets_estimated}]: {target_localized}\n")

            # 二乗誤差の算出
            squared_error = distance_error_squared.calculate(target, target_localized)
            squared_error_list = np.append(squared_error_list, squared_error)

            if squared_error >= 5.0:
              # print("\n")
              # print(f"{is_positive_probable}"+f" / {squared_error}"+f" / {recursive_count}")
              # センサーの位置（灰色）・測位可能なセンサーの位置（黒色）・アンカーノードの位置（オレンジ）・測位可能なセンサーの位置（緑）ターゲットの推定位置（青）・ターゲットの位置（赤）
              # plt.scatter(sensors[:, 0], sensors[:, 1], c="gray")
              # plt.scatter(sensors_available_for_target_estimated_orignal[:, 0], sensors_available_for_target_estimated_orignal[:, 1], c="black")
              # plt.scatter(anchors[:, 0], anchors[:, 1], c="orange")
              # plt.scatter(sensors_available_for_target_estimated[:, 0], sensors_available_for_target_estimated[:, 1], c="green")
              # plt.scatter(target_estimated[0], target_estimated[1], c="blue")
              # plt.scatter(target[0], target[1], c="red")
              # plt.show()
              # plt.close('all')
              # plt.clf()              
              big_error_count += 1
            
          
              


            # 協調測位であれば測位したTNをSNに追加する（RNに変更する）
            if is_cooperative:
              sensors_original = np.append(sensors_original, [target], axis=0)
              sensors = np.append(sensors, [target_localized], axis=0)

            # 測位フラグの更新
            targets[index_targets_estimated, 2] = 1
          
          else:
            targets_unlocalized_count[index_targets_estimated] += 1
        
        if is_successive:
          index_targets_begin = np.max(mask_targets_estimated) + 1

      is_localizable = np.any(targets_unlocalized_count[mask_targets_unlocalized] < max_localization_loop)
      # if not is_localizable:
        # print(f"\ntargets:\n {targets}")
        # print(f"unlocalized count:\n{targets_unlocalized_count}")
        # print(f"unlocalized count:\n{squared_error_list}")

      if len(targets_localized) == targets_count:
        break

    # if recursive_count < 3:
    #   print("\n")
    #   print(f"{is_positive_probable}"+f" / {squared_error}"+f" / {recursive_count}")
    
    # targets_unlocalized = targets[targets[:, 2] == 0]
    # if len(targets_unlocalized) > 0 and np.any(np.linalg.norm(targets_unlocalized[:2] - 15.0, axis=0) <= 2.0):
    #   plt.scatter(sensors[:, 0], sensors[:, 1], c="gray")
    #   plt.scatter(sensors_available_for_target_estimated_orignal[:, 0], sensors_available_for_target_estimated_orignal[:, 1], c="black")
    #   plt.scatter(anchors[:, 0], anchors[:, 1], c="orange")
    #   plt.scatter(sensors_available_for_target_estimated[:, 0], sensors_available_for_target_estimated[:, 1], c="green")
    #   plt.scatter(target_estimated[0], target_estimated[1], c="blue")
    #   plt.scatter(target[0], target[1], c="red")
    #   plt.show()
    #   plt.close('all')
    #   plt.clf()

    # 測位時間の算出
    time_localization_end = time.time()
    duration_localization_per_target =  (time_localization_end - time_localization_start)/(len(targets_localized) or 1)
    if sim_cycle == 0:
      duration_localization_per_target_avg = duration_localization_per_target
    else:
      duration_localization_per_target_avg = (duration_localization_per_target_avg*sim_cycle + duration_localization_per_target)/(sim_cycle + 1)

    # シミュレーション全体におけるMSE及びRMSEの算出
    squared_error_total += np.sum(squared_error_list)
    # squared_error_total += np.nansum(squared_error_list)
    # if len(targets_localized) == 0:
    #   print("測位がうまくいっていません")
    targets_localized_count_total += len(targets_localized)
    mean_squared_error = squared_error_total/targets_localized_count_total
    root_mean_squared_error = np.sqrt(mean_squared_error)
    
    # 求めたRMSEをリストに追加
    root_mean_squared_error_list = np.append(root_mean_squared_error_list, root_mean_squared_error)

    # RMSE（シミュレーション平均）の算出
    if sim_cycle == 0:
      root_mean_squared_error_avg = root_mean_squared_error
    else:
      root_mean_squared_error_avg = (root_mean_squared_error_avg*sim_cycle + root_mean_squared_error)/(sim_cycle + 1)
    
    # RMSEの分布を更新（協調測位の場合はRMSEの値が大きく振れるのであまり意味がないかも）
    # field_rmse_distribution = rmse_distribution.update(field_rmse_distribution, grid_interval, targets_localized, squared_error_list)

    # 測位順と測位誤差のリスト
    # squared_error_lists = np.append(squared_error_lists, np.array([squared_error_list]), axis=0)
    # if np.any(np.linalg.norm(targets_localized[:, :2] - centroid_of_anchors, axis=1)>35):
      # targets_plot = np.append(targets,np.abs(targets[:, :2]))
      # print(f"{targets}")
      # print("__")
      # # センサーの位置（灰色）・測位可能なセンサーの位置（黒色）・アンカーノードの位置（オレンジ）・測位可能なセンサーの位置（緑）ターゲットの推定位置（青）・ターゲットの位置（赤）
      # plt.scatter(sensors[:, 0], sensors[:, 1], c="gray")
      # plt.scatter(sensors_available_for_target_estimated_orignal[:, 0], sensors_available_for_target_estimated_orignal[:, 1], c="black")
      # plt.scatter(anchors[:, 0], anchors[:, 1], c="orange")
      # plt.scatter(sensors_available_for_target_estimated[:, 0], sensors_available_for_target_estimated[:, 1], c="green")
      # plt.scatter(target_estimated[0], target_estimated[1], c="blue")
      # plt.scatter(target[0], target[1], c="red")
      # plt.show()
      # plt.close('all')
      # plt.clf()

    # 測位可能確率の分布の更新とその平均の算出
    field_localizable_probability_distribution = localizable_probability_distribution.update(field_localizable_probability_distribution, grid_interval, targets, targets_localized)
    localizable_probability_avg = np.sum(field_localizable_probability_distribution[:, 2]*field_localizable_probability_distribution[:, 3])/np.sum(field_localizable_probability_distribution[:, 3])

    print("\r" + "{:.3f}".format((sim_cycle + 1)/sim_cycles*100) + "%" + " done." + " / RMSE: " + "{:.4f}".format(root_mean_squared_error_avg) + " / Avg. Localizable Prob.: " + "{:.4f}".format(localizable_probability_avg) + " / Avg. Localization Duration per target: " + "{:.6f}".format(duration_localization_per_target_avg), end="")
    print(" / avg_big_error_count: " + "{:.4f}".format((big_error_count)/(sim_cycle+1)), end="")
    

    if is_recursive:
      if sim_cycle == 0:
        recursive_count_avg = recursive_count
      else:
        recursive_count_avg = (recursive_count_avg*sim_cycle + recursive_count)/(sim_cycle + 1)
      print(" / Avg. Recursive Count: " + "{:.4f}".format(recursive_count_avg), end="")

  print("\n")
  sim_time_end = time.time()
  sim_time_length = sim_time_end - sim_time_start
  #シミュレーションの開始・終了で平均シミュレーション時間を計算
  sim_time_ave = sim_time_length/sim_cycles
  
  print(f"RMSE: {root_mean_squared_error_avg} m")

  # 結果を出力
  result_data = pd.DataFrame({
    "RMSE": [root_mean_squared_error_avg],
    "Avg. Localization Prob.": [localizable_probability_avg],
    "Avg. Localization Duration per target": [duration_localization_per_target_avg]
  })
  if is_recursive:
    result_data["Avg. Recursive Count"] = [recursive_count_avg]
  result_filename = "result.csv"
  result_filepath = os.path.join(output_dirpath, result_filename)
  result_data.to_csv(result_filepath, index=False)
  print(f"result was saved in {result_filepath}")

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

  print("\ncomplete.")

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