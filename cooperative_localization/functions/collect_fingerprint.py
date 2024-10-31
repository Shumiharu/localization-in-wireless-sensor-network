import os
import sys
import numpy as np
import yaml
import pandas as pd
import csv

# 基本関数
from basis import distance_toa

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

  # Field Config
  field_range = config["field_range"]
  x_top, x_bottom, y_top, y_bottom = field_range["x_top"], field_range["x_bottom"], field_range["y_top"], field_range["y_bottom"]
   
  center_of_field = np.array([np.mean(np.array([x_top, x_bottom])), np.mean(np.array([y_top, y_bottom]))])
  grid_interval = field_range["grid_interval"]
  x_range = np.arange(x_bottom, x_top + grid_interval, grid_interval)
  y_range = np.arange(y_bottom, y_top + grid_interval, grid_interval)
  field_rmse_distribution = np.array([[x, y, 0.0, 0] for x in x_range for y in y_range]) # rmseの分布を算出 -> [x, y, rmse, data_count]
  field_localizable_probability_distribution = np.copy(field_rmse_distribution) # 測位可能確立の分布を算出 -> [x, y, localizable probability, data_count]

  width = x_top - x_bottom
  height = y_top - y_bottom
  print(f"field: {width} x {height}")

  # Anchors & Targets Config
  anchors_config = config["anchors"]
  anchors = np.array([[anchor_config["x"], anchor_config["y"], 1] for anchor_config in anchors_config])
  print("anchor: (x, y) = ", end="")
  for anchor_config in anchors_config:
    anchor_x = anchor_config["x"]
    anchor_y = anchor_config["y"]
    print(f"({anchor_x}, {anchor_y})", end=" ")
  print(f"\n=> anchor count: {len(anchors_config)}")

  targets = np.array([[x_top*i/8.0, y_top*j/8.0] for i in [1, 3, 5, 7] for j in [1, 3, 5, 7]])
  
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

  # Fingerprint
  fingerprint_count = config["fingerprint"]["count"]
  fingerprint_filename = config["fingerprint"]["filename"]
  fingerprint_filepath = "../fingerprint/" + fingerprint_filename
  print(f"{fingerprint_filename} will be saved in {fingerprint_filepath}")

  # Temporary Prameter
  rx_power_list = np.zeros((len(targets), len(anchors)))
  fingerprint_count_list = np.copy(rx_power_list)
  fingerprint_count_max = fingerprint_count_list.size*fingerprint_count
  
  print("\n")

  # シミュレーション開始
  while np.any(fingerprint_count_list < fingerprint_count):
    for i, target_measurement in enumerate(targets):
      for j, anchor in enumerate(anchors):
        if fingerprint_count_list[i, j] < fingerprint_count:
          distance_accurate = np.linalg.norm(target_measurement[:2] - anchor[:2])
          distance_measured, rx_power = distance_toa.calculate(channel, max_distance_measurement, distance_accurate)
          if not np.isinf(distance_measured):
            rx_power_list[i, j] = (rx_power_list[i, j]*fingerprint_count_list[i, j] + rx_power)/(fingerprint_count_list[i, j] + 1)
            fingerprint_count_list[i, j] += 1
    
    progress = (np.sum(fingerprint_count_list)/fingerprint_count_max)*100
    print("\r" + "{:.3f}".format(progress) + "%" + " done.", end="")
  

  
  targets_data = pd.DataFrame({
    "x": targets[:, 0],
    "y": targets[:, 1]
  })

  rx_power_data = pd.DataFrame({
    f"anchor({anchor[0]} {anchor[1]})_rx_power": rx_power_list[:, i]
    for i, anchor in enumerate(anchors)
  })

  fingerprint_data = pd.concat([targets_data, rx_power_data], axis=1)
  fingerprint_data.to_csv(fingerprint_filepath, index=False, quoting=csv.QUOTE_NONE, escapechar=',')
  print(f"\n{fingerprint_filename} was saved in {fingerprint_filepath}")

  print("\ncomplete.")

# 
# targets_measurement = np.array([[x_top*i/8.0, y_top*j/8.0] for i in [3, 5, 7] for j in [3, 5, 7]])
# mask_targets_unique = np.ones(len(targets), dtype=bool)
# for target_measurement in targets_measurement:
#   mask_targets_unique &= ~np.all(targets == target_measurement, axis=1)
# targets_complement = targets[mask_targets_unique]

# 対称点に対応する受信電力の生成
# indices = [8, 6, 7, 2, 4, 5, 0]
# rx_power_list_complement = np.array([[rx_power_list[index, 3], rx_power_list[index, 2], rx_power_list[index, 1], rx_power_list[index, 0]] for index in indices])
# rx_power_list = np.append(rx_power_list, rx_power_list_complement, axis=0)
# targets = np.append(targets_measurement, targets_complement, axis=0)