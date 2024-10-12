import numpy as np
import pandas as pd
import yaml
import joblib

from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
  # Open configuration file
  config_filename = "config_0.yaml"
  config_filepath = "configs/" + config_filename
  with open(config_filepath, "r") as config_file:
    config = yaml.safe_load(config_file)
    print(f"{config_filename} was loaded")

  # サンプルデータの読み出し
  sample_filename = "sample_1.csv"
  sample_filepath = "samples/" + sample_filename
  features_data = pd.read_csv(sample_filepath)
  features_list = features_data.to_numpy()
  print(f"{sample_filename} was loaded.")
  
  # モデルの読み出し
  model_filename = "model_0.pkl"
  model_filepath = "models/" + model_filename
  model = joblib.load(model_filepath)

  error_threshold = config["model"]["error_threshold"]
  labels = np.where(features_list[:, -1] >= error_threshold, 1, 0)
  predicted = model.predict(features_list[:, :-1])

  print(f"accuracy_score: {accuracy_score(y_true=labels, y_pred=predicted)}")
  print(f"recall_score: {recall_score(y_true=labels, y_pred=predicted)}")