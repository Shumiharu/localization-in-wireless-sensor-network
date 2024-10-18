import os
import sys
import time
import yaml
import subprocess
import numpy as np
from datetime import datetime

def usage():
  print('Usage: python launch.py <config> (referenced at ./configs)')
  exit()

# 複数のプログラムを同時に実行するプログラム
if __name__ == "__main__":
  
  current_dirpath = os.getcwd()
  args = sys.argv
  if not len(args) == 2:
    usage()

  # Open configuration file
  config_filename = args[1] + ".yaml"
  config_filepath = os.path.join("configs", config_filename)
  with open(config_filepath, "r") as config_file:
    config = yaml.safe_load(config_file)
  
  # Make Folder and Save Config
  now = datetime.now()
  output_dirname = now.strftime("%Y-%m-%d_%H-%M-%S")
  output_dirpath = os.path.join("output", output_dirname)
  os.makedirs(output_dirpath, exist_ok=True)
  print(f"{output_dirname} was made in {output_dirpath}.")

  config_referenced_filename = "config.yaml"
  config_referenced_filepath = os.path.join(output_dirpath, config_referenced_filename)
  with open(config_referenced_filepath, "w") as config_referenced_file:
    yaml.safe_dump(config, config_referenced_file)
    print(f"This program's config references at {config_referenced_filepath}")
  
  # Cooperative Localization or not
  is_cooperative_localization = config["localization"]["is_cooperative"]
  print("Localization: Least Square (LS) Method", end=" ")
  print("with Cooperation" if is_cooperative_localization else "without Cooperation")
  
  # 機械学習を利用しない場合は1~4のステップを省略
  is_predictive = config["localization"]["is_predictive"]
  if is_predictive:
    # 1. Collect Sample Data
    try:
      command_collect_sample_data = f"cd functions && python collect_sample_data.py ../{output_dirpath}"
      print('\033[36m' + f'{current_dirpath}' + '\033[0m' + f'$ {command_collect_sample_data}')
      # process_collect_sample_data = subprocess.Popen(command_collect_sample_data, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      process_collect_sample_data = subprocess.Popen(command_collect_sample_data, shell=True)
    except subprocess.CalledProcessError as e:
      print(e)
      exit()
    print("Collecting sample data...")

    # 2. Collect Evaluation Data
    try:
      command_collect_evaluation_data = f"cd functions && python collect_evaluation_data.py ../{output_dirpath}"
      print('\033[36m' + f'{current_dirpath}' + '\033[0m' + f'$ {command_collect_evaluation_data}')
      # process_collect_evaluation_data = subprocess.Popen(command_collect_evaluation_data, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      process_collect_evaluation_data = subprocess.Popen(command_collect_evaluation_data, shell=True)
    except subprocess.CalledProcessError as e:
      print(e)
      exit()
    print("Collecting Evaluation data...")
    
    process_collect_sample_data.wait()
    print("\nSample data was collected.")

    # 3. Build Model
    try:
      command_build_model = f"cd functions && python build_model.py ../{output_dirpath}"
      print('\033[36m' + f'{current_dirpath}' + '\033[0m' + f'$ {command_build_model}')
      process_build_model = subprocess.run(command_build_model, shell=True, check=True)
    except subprocess.CalledProcessError as e:
      print(e)
      exit()
    print("\nModel was built.")

    process_collect_evaluation_data.wait()
    print("\nEvaluation data was collected.")
    
    # 4. Evaluate Model
    try:
      command_evaluate_model = f"cd functions && python evaluate_model.py ../{output_dirpath}"
      print('\033[36m' + f'{current_dirpath}' + '\033[0m' + f'$ {command_evaluate_model}')
      subprocess.run(command_evaluate_model, shell=True, check=True)
    except subprocess.CalledProcessError as e:
      print(e)
      exit()

    print("\nModel was Evaluated.")
  
  # Localization
  try:
    command_localize = f"cd functions && python localize.py ../{output_dirpath}"
    print('\033[36m' + f'{current_dirpath}' + '\033[0m' + f'$ {command_localize}')
    subprocess.run(command_localize, shell=True, check=True)
  except subprocess.CalledProcessError as e:
    print(e)
    exit()
  print("\ncomplete.")