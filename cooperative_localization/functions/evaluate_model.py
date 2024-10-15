import os
import sys
import numpy as np
import pandas as pd
import yaml
import joblib

from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

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

  # Read Evaluation Data File
  evaluation_data_filename = config["evaluation_data"]["filename"]
  evaluation_data_filepath = "../evaluation_data/" + evaluation_data_filename
  features_evaluation_data = pd.read_csv(evaluation_data_filepath)
  features_evaluation_list = features_evaluation_data.to_numpy()
  print(f"{evaluation_data_filename} was loaded from {evaluation_data_filepath}")
  
  # Load Model File
  model_filename = config["model"]["filename"]
  model_filepath = "../models/" + model_filename
  model = joblib.load(model_filepath)
  print(f"{model_filename} was loaded from {model_filepath}.")

  error_threshold = config["model"]["error_threshold"]
  # labels = np.where(features_evaluation_list[:, -1] >= error_threshold, 1, 0)
  labels = features_evaluation_list[:, -1]
  predicted = model.predict(features_evaluation_list[:, :-1])

  print(f"accuracy_score: {accuracy_score(y_true=labels, y_pred=predicted)}")
  print(f"recall_score: {recall_score(y_true=labels, y_pred=predicted)}")