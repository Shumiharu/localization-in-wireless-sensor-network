import os
import sys
import numpy as np
import pandas as pd
import yaml
import joblib

import lightgbm as lgb

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

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

  # サンプルデータの読み出し
  sample_data_filename = config["sample_data"]["filename"]
  sample_data_filepath = "../sample_data/" + sample_data_filename
  features_data = pd.read_csv(sample_data_filepath)
  features_sample_list = features_data.to_numpy()
  print(f"{sample_data_filename} was loaded.")

  # Model
  model_type = config["model"]["type"]
  model_filename = config["model"]["filename"]
  model_filepath = "../models/" + model_filename
  error_threshold = config["model"]["error_threshold"]

  # 正解ラベル
  labels = np.where(features_sample_list[:, -1] >= error_threshold, 1, 0)
  # labels = features_sample_list[:, -1] # 山本先輩結果合わせのため

  # 学習用と評価をランダムに抽出
  explanatory_variables_train, explanatory_variables_test, lables_train, lables_test = train_test_split(features_sample_list[:, :-1], labels, stratify=labels, random_state=0)

  # SVMのパイプラインを作成
  #pipe_line = make_pipeline(StandardScaler(), SVC(random_state=0))

  #ランダムフォレスト版
  #pipe_line = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0))

  #lightgbm(勾配ブースティング木)版
  #pipe_line = make_pipeline(StandardScaler(), lgb.LGBMClassifier(random_state=0))

  #neural network版
  pipe_line = make_pipeline(StandardScaler(),MLPClassifier(activation='logistic',random_state=0))

  # Cパラメータの設定
  cost_parameter_range = config["model"]["cost_parameter_range"]
  #cost_parameter_grid = [{"svc__C": cost_parameter_range, "svc__kernel": ["rbf"]}]

  #ランダムフォレスト版
  #cost_parameter_grid = [{'randomforestclassifier__n_estimators':[50,100,150],'randomforestclassifier__max_depth':[5,10,15]}]
  
  #lightgbm(勾配ブースティング木)版
  #cost_parameter_grid = [{'lgbmclassifier__n_estimators':[5,10,15],'lgbmclassifier__max_depth':[5,10,15] }]

  #neural network版
  cost_parameter_grid = [{"mlpclassifier__learning_rate_init":[0.01,0.005,0.001]}]

  # グリッドサーチ
  grid_search = GridSearchCV(
    estimator = pipe_line,
    param_grid = cost_parameter_grid,
    scoring = "accuracy",
    cv = 5,
    refit = True,
    n_jobs= -1,
    error_score="raise"
  ).fit(explanatory_variables_train, lables_train)
  print(f"grid_search.best_score_: {grid_search.best_score_}")
  print(f"grid_search.best_params_: {grid_search.best_params_}")

  # 最適パラメータを用いてモデルを作成・保存
  model = grid_search.best_estimator_.fit(explanatory_variables_train, lables_train)
  joblib.dump(model, model_filepath)

  print(f"{model_filename} was built in {model_filepath}")

