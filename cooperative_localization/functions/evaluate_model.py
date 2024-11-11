import os
import sys
import numpy as np
import pandas as pd
import yaml
import joblib


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


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

  # Learning Model
  is_built_successively = config["model"]["is_built_successively"]
  is_models_example = config["model"]["is_example"]
  model_subdirname = "successive" if is_built_successively else "collective"
  model_type = config["model"]["type"]
  model_filename = config["model"]["filename"]
  model_filepath = f"../models/{model_subdirname}/{model_type}/{model_filename}"
  if is_models_example:
    model_filepath = f"../models_example/{model_subdirname}/{model_type}/{model_filename}"
  model = joblib.load(model_filepath)
  print(f"\n{model_filename} was loaded from {model_type} -> {model_filepath}.")

  # Read Evaluation Data File
  is_evaluation_data_example = config["evaluation_data"]["is_example"]
  evaluation_data_subdirname = model_subdirname
  evaluation_data_filename = config["evaluation_data"]["filename"]
  evaluation_data_filepath = f"../evaluation_data/{evaluation_data_subdirname}/{evaluation_data_filename}"
  if is_evaluation_data_example:
    evaluation_data_filepath = f"../evaluation_data_example/{evaluation_data_subdirname}/{evaluation_data_filename}"
  features_evaluation_data = pd.read_csv(evaluation_data_filepath)
  features_evaluation_list = features_evaluation_data.to_numpy()
  print(f"\n{evaluation_data_filename} was loaded from {evaluation_data_filepath}")
  
  error_threshold = config["model"]["error_threshold"]
  labels = np.where(features_evaluation_list[:, -1] >= error_threshold, 1, 0)
  # labels = features_evaluation_list[:, -1] # 山本先輩結果合わせのため
  predicted = model.predict(features_evaluation_list[:, :-1])

  accuracy = accuracy_score(y_true=labels, y_pred=predicted)
  precision = precision_score(y_true=labels, y_pred=predicted)
  recall = recall_score(y_true=labels, y_pred=predicted)

  print(f"accuracy_score: {accuracy}")
  print(f"precision_score: {precision}")
  print(f"recall_score: {recall}")

  if is_subprocess:
    output_dirpath = args[1]
    score_data = pd.DataFrame({
      "type": [model_type],
      "model_filepath": [model_filepath],
      "accuracy_score": [accuracy],
      "precision_score": [precision],
      "recall_score": [recall]
    })
    score_data_filepath = os.path.join(output_dirpath, 'model_score.csv')
    score_data.to_csv(score_data_filepath, index=False)
    print(f"model_score.csv was saved in {score_data_filepath}.")