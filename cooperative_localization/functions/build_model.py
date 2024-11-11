import os
import sys
import numpy as np
import pandas as pd
import yaml
import joblib
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve

def plot_validation_curve(model, explanatory_variables_train, labels_train, param_name, param_range, param_scales, cv, scoring, best_param_):
    
    # param_rangeにNone，もしくは文字列が含まれる場合，以下の処理は行わない
    if any(isinstance(item, str) or item is None for item in param_range):
      return
    
    train_scores, valid_scores = validation_curve(
      estimator=model,
      X=explanatory_variables_train,
      y=labels_train,
      param_name=param_name,
      param_range=param_range,
      cv=cv,
      scoring=scoring,
      n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    
    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    
    plt.plot(param_range, valid_mean, color='green', linestyle='--', marker='o', markersize=5, label='Validation score')
    plt.fill_between(param_range, valid_mean + valid_std, valid_mean - valid_std, alpha=0.15, color='green')
    
    if best_param_ == "" or best_param_ is not None:
      plt.axvline(x=best_param_, color='gray')

    plt.xscale(param_scales)

    plt.xlabel(param_name)
    plt.ylabel(scoring)

    plt.legend(loc='best')

    plt.grid()
    plt.show()

def plot_learning_curve(model, explanatory_variables_train, labels_train, cv, scoring):
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=model,
        X=explanatory_variables_train,
        y=labels_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    # 学習データ指標の平均±標準偏差を計算
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    train_center = train_mean
    train_high = train_mean + train_std
    train_low = train_mean - train_std
    # 検証データ指標の平均±標準偏差を計算
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std  = np.std(valid_scores, axis=1)
    valid_center = valid_mean
    valid_high = valid_mean + valid_std
    valid_low = valid_mean - valid_std
    # training_scoresをプロット
    plt.plot(train_sizes, train_center, color='blue', marker='o', markersize=5, label='training score')
    plt.fill_between(train_sizes, train_high, train_low, alpha=0.15, color='blue')
    # validation_scoresをプロット
    plt.plot(train_sizes, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
    plt.fill_between(train_sizes, valid_high, valid_low, alpha=0.15, color='green')
    # 最高スコアの表示
    best_score = valid_center[len(valid_center) - 1]
    plt.text(np.amax(train_sizes), valid_low[len(valid_low) - 1], f'best_score={best_score}',
                    color='black', verticalalignment='top', horizontalalignment='right')
    # 軸ラベルおよび凡例の指定
    plt.xlabel('training examples')  # 学習サンプル数を横軸ラベルに
    plt.ylabel(scoring)  # スコア名を縦軸ラベルに
    plt.legend(loc='lower right')  # 凡例
    plt.grid()
    plt.show()

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
  is_plot_curves = config["model"]["is_plot_curves"]
  
  model_subdirname = "successive" if is_built_successively else "collective"
  model_type = config["model"]["type"]
  model_filename = config["model"]["filename"]
  model_filepath = f"../models/{model_subdirname}/{model_type}/{model_filename}"
  print(f"\nModel will be built in {model_filepath}")

  # Read Sample Data
  is_sample_data_example = config["sample_data"]["is_example"]
  sample_data_subdirname = model_subdirname
  sample_data_filename = config["sample_data"]["filename"]
  sample_data_filepath = f"../sample_data/{sample_data_subdirname}/{sample_data_filename}"
  if is_sample_data_example:
    sample_data_filepath = f"../sample_data_example/{sample_data_subdirname}/{sample_data_filename}"
  features_data = pd.read_csv(sample_data_filepath)
  features_sample_list = features_data.to_numpy()
  print(f"\n{sample_data_filename} was loaded from {sample_data_filepath}")

  # 正解ラベル
  error_threshold = config["model"]["error_threshold"]
  labels = np.where(features_sample_list[:, -1] >= error_threshold, 1, 0)
  # labels = features_sample_list[:, -1] # 山本先輩結果合わせのため

  # 学習用と評価をランダムに抽出
  explanatory_variables_train, explanatory_variables_test, labels_train, labels_test = train_test_split(features_sample_list[:, :-1], labels, stratify=labels, random_state=0)

  # 交差検証（クロスバリデーション）の設定
  cv = KFold(n_splits=5, shuffle=True, random_state=42)
  # cv = 5

  # 評価指標の設定
  scoring = "accuracy"

  # support vector machine
  if model_type == "svm":
    pipe_line = make_pipeline(StandardScaler(), SVC(random_state=0))
    param_config = {
      'svc__C': (config["model"]["cost_parameter_range"], 'linear'),
      'svc__kernel': (['rbf'], 'linear')
    }
  
  # random forest
  if model_type == "rf":
    pipe_line = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0))
    param_config = {
      'randomforestclassifier__n_estimators': ([50, 75, 100, 125], 'linear'), # 100
      # 'randomforestclassifier__criterion': (["gini", "entropy"], 'linear'),
      'randomforestclassifier__max_depth':([15, 20, 25], 'linear'), # 25
      'randomforestclassifier__min_samples_split': ([10, 12, 14], 'linear'), # 12
      # 'randomforestclassifier__max_leaf_nodes': ([None, 10, 30], 'linear'),
    }
  
  # light gbm （勾配ブースティング木）
  if model_type == "lgb":
    pipe_line = make_pipeline(StandardScaler(), lgb.LGBMClassifier(random_state=0))
    param_config = {
      # 'lgbmclassifier__n_estimators': ([5,10,15], 'linear'),
      'lgbmclassifier__reg_alpha': ([0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10], 'log'),
      'lgbmclassifier__reg_lambda': ([0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10], 'log'),
      'lgbmclassifier__num_leaves': ([2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256], 'linear'),
      'lgbmclassifier__colsample_bytree': ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'linear'),
      'lgbmclassifier__subsample': ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'linear'),
      'lgbmclassifier__subsample_freq': ([0, 1, 2, 3, 4, 5, 6, 7], 'linear'),
      'lgbmclassifier__min_child_samples': ([0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'linear'),
    }

  # xgboost （勾配ブースティング木）
  if model_type == "xgb":
    pipe_line = make_pipeline(StandardScaler(), xgb.XGBClassifier(random_state=0))
    param_config = {
      'xgbclassifier__n_estimators': ([50, 70, 90, 100, 110, 120], 'linear'), # 100
      'xgbclassifier__colsample_bytree': ([1.0], 'linear'),
      'xgbclassifier__gamma': ([1], 'log'),
      'xgbclassifier__learning_rate': ([0.1], 'log'),
      'xgbclassifier__max_depth': ([4], 'linear'),
      'xgbclassifier__min_child_weight': ([5], 'linear'),
      'xgbclassifier__subsample': ([0.4], 'linear'),
    }

  # neural network
  if model_type == "nn":
    pipe_line = make_pipeline(StandardScaler(),MLPClassifier(activation='logistic',random_state=0, max_iter=500, early_stopping=True))
    param_config = {
      'mlpclassifier__learning_rate_init': ([0.001, 0.005, 0.01], 'log')
    }
    # param_grid = [{"mlpclassifier__learning_rate_init":[0.01,0.005,0.001]}]

  param_grid = {key: value[0] for key, value in param_config.items()}
  param_scale = {key: value[1] for key, value in param_config.items()}

  # 
  if is_plot_curves:
    # チューニング前のモデルの作成
    model = pipe_line.fit(
      explanatory_variables_train,
      labels_train
    )

    # 検証曲線の表示
    for key, value in param_config.items():
      plot_validation_curve(model=model,
                            explanatory_variables_train=explanatory_variables_train,
                            labels_train=labels_train,
                            param_name=key,
                            param_range=value[0],
                            param_scales=value[1],
                            cv=cv,
                            scoring=scoring,
                            best_param_=""
      )

  # グリッドサーチ実行
  gridcv = GridSearchCV(
    estimator = pipe_line,
    param_grid = param_grid,
    scoring = scoring,
    cv=cv,
    refit = True,
    n_jobs= -1,
    error_score="raise"
  )
  gridcv.fit(
    explanatory_variables_train,
    labels_train
  )

  best_params_ = gridcv.best_params_
  best_score_ = gridcv.best_score_
  print(f"gridcv.best_params_: {best_params_}")
  print(f"gridcv.best_score_: {best_score_}")

  if is_subprocess:
    output_dirpath = args[1]
    tuning_data = pd.DataFrame({
      "type": [model_type],
      "model_filepath": [model_filepath],
    })
    for key, value in gridcv.best_params_.items():
      tuning_data[key] = [value]
    tuning_data["best_score_"] = [best_score_]
    tuning_data_filepath = os.path.join(output_dirpath, 'tuning_data.csv')
    tuning_data.to_csv(tuning_data_filepath, index=False)
    print(f"tuning_data.csv was saved in {tuning_data_filepath}.")

  # 最適パラメータを用いてモデルを作成・保存
  model = gridcv.best_estimator_.fit(explanatory_variables_train, labels_train)
  
  if is_plot_curves:
    plot_learning_curve(
      model=model,
      explanatory_variables_train=explanatory_variables_train,
      labels_train=labels_train,
      cv=cv,
      scoring=scoring
    )
    for key, value in param_config.items():
      best_param_ = best_params_[key]
      plot_validation_curve(
        model=model,
        explanatory_variables_train=explanatory_variables_train,
        labels_train=labels_train,
        param_name=key,
        param_range=value[0],
        param_scales=value[1],
        cv=cv,
        scoring=scoring,
        best_param_=best_param_
      )

  joblib.dump(model, model_filepath)
  print(f"{model_filename} was built in {model_filepath}")

# メモ
# https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359
# https://qiita.com/c60evaporator/items/a9a049c3469f6b4872c6
# に従って作成しました