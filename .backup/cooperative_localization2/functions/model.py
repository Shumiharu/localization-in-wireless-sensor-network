import numpy as np
import pandas as pd
import yaml
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

def build(sample_data: np.ndarray, labels: np.ndarray, cost_parameter_range):

  # 学習用と評価用をランダムに抽出
  explanatory_variables_train, explanatory_variables_test, lables_train, lables_test = train_test_split(sample_data[:, :-1], labels, stratify=labels, random_state=0)

  # SVMのパイプラインを作成
  pipe_line = make_pipeline(StandardScaler(), SVC(random_state=0))

  # Cパラメータの設定
  cost_parameter_grid = [{"svc__C": cost_parameter_range, "svc__kernel": ["rbf"]}]

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

  # print(grid_search.best_score_)
  # print(grid_search.best_params_)

  # 最適パラメータを用いてモデルを作成・保存
  model = grid_search.best_estimator_.fit(explanatory_variables_train, lables_train)

  return model

def evaluate(model, sample_data: np.ndarray, labels: np.ndarray) -> list:
  # labels = np.where(sample_data[:, -1] >= error_threshold, 1, 0)
  predicted = model.predict(sample_data[:, :-1])
  return [accuracy_score(y_true=labels, y_pred=predicted), recall_score(y_true=labels, y_pred=predicted)]

# Example Usage
if __name__ == "__main__":
  config_filename = "config_0.yaml"
  config_filepath = "../configs/" + config_filename
  with open(config_filepath, "r") as config_file:
    config = yaml.safe_load(config_file)
    print(f"{config_filename} was loaded")
  
  error_threshold = config["model"]["error_threshold"]
  cost_parameter_range = config["model"]["cost_parameter_range"]
  
  # サンプルデータの読み出し
  sample_filename = "sample_0.csv"
  sample_filepath = "../samples/" + sample_filename
  sample_data = pd.read_csv(sample_filepath).to_numpy()
  print(f"{sample_filename} was loaded.")
  
  # 正解ラベル
  labels = np.where(sample_data[:, -1] >= error_threshold, 1, 0)

  # モデルを作成
  model = build(sample_data, labels, cost_parameter_range)
  model_filename = "model_0.pkl"
  model_filepath = "../models/" + model_filename
  joblib.dump(model, model_filepath)
  print(f"{model_filename} was built in {model_filepath}")

  # スコアを算出
  score = evaluate(model, sample_data, labels)
  print(f"accuracy: {score[0]}, recall: {score[1]}")



