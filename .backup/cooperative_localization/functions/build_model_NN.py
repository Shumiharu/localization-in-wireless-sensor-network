import os
import sys
import numpy as np
import pandas as pd
import yaml
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

def create_model(optimizer='adam'):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(explanatory_variables_train.shape,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
  
  args = sys.argv
  is_subprocess = True if len(args) == 2 else False

  # Open configuration file
  config_filename = "config_0.yaml"
  config_filepath = "../configs/" + config_filename
  if is_subprocess:
    config_filepath = os.path.join(args, "config.yaml")
  with open(config_filepath, "r") as config_file:
    config = yaml.safe_load(config_file)
    print(f"{config_filename} was loaded from {config_filepath}")

  # サンプルデータの読み出し
  sample_data_filename = config["sample_data"]["filename"]
  sample_data_filepath = "../sample_data/" + sample_data_filename
  features_data = pd.read_csv(sample_data_filepath)
  features_list = features_data.to_numpy()
  print(f"{sample_data_filename} was loaded.")

  # Model
  model_filename = config["model"]["filename"]
  model_filepath = "../models/" + model_filename
  error_threshold = config["model"]["error_threshold"]

  # 正解ラベル
  labels = np.where(features_list[:, -1] >= error_threshold, 1, 0)

  # 学習用と評価をランダムに抽出
  explanatory_variables_train, explanatory_variables_test, labels_train, labels_test = train_test_split(features_list[:, :-1], labels, stratify=labels, random_state=0)

  # データの標準化
  scaler = StandardScaler()
  explanatory_variables_train = scaler.fit_transform(explanatory_variables_train)
  explanatory_variables_test = scaler.transform(explanatory_variables_test)

  # KerasClassifierのラッパーを使用してモデルを作成
  model = KerasClassifier(build_fn=create_model, verbose=0)
  model = Sci-Keras(build_fn=create_model, verbose=0)
  

  # グリッドサーチのパラメータ設定
  param_grid = {
      'batch_size': [5, 10, 20],
      'epochs': [50, 100],
      'optimizer': ['adam', 'rmsprop']
  }

  # グリッドサーチの実行
  grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, error_score='raise')
  grid_result = grid.fit(explanatory_variables_train, labels_train)

  # 最適パラメータの表示
  print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

  # 最適モデルの保存
  best_model = grid_result.best_estimator_.model
  best_model.save(model_filepath)

  # モデルの評価
  loss, accuracy = best_model.evaluate(explanatory_variables_test, labels_test)
  print(f"Test accuracy: {accuracy}")

  print(f"{model_filename} was built in {model_filepath}")
