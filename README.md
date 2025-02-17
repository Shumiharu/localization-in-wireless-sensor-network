# Cooperative Localization

## ディレクトリ構成
```
.
├── cooperative_localization
│   ├── configs # このconfigのパラメータをlaunch.pyやlocalize.pyが読み出してシミュレーションが実行される
│   │   ├── config_0.yaml # 基本はここのパラメータを変更する
│   │   └── config_dbg.yaml # デバッグ用
│   ├── evaluation_data # モデルの性能を検証するデータ（evaluate_model.pyで出力されたデータの保存先）
│   │   ├── collective # 各TNが取得した測距値を元にAPサーバーがまとめて測位する清水手法
│   │   └── successive # 各TNに対してAPサーバーが逐次的に測位する山本手法
│   ├── evaluation_data_example # 必要に応じて追加したい検証データをここへ
│   │   ├── collective # 各TNが取得した測距値を元にAPサーバーがまとめて測位する清水手法
│   │   └── successive # 各TNに対してAPサーバーが逐次的に測位する山本手法
│   ├── fingerprint # たぶん今後は使わない
│   ├── fingerprint_example # たぶん今後は使わない
│   ├── functions 
│   │   ├── basis # LS法を用いた測位の基本的な関数群（詳細は各プログラムファイルを参照）
│   │   │   ├── awgn.py
│   │   │   ├── distances_avg.py
│   │   │   ├── distance_toa.py
│   │   │   ├── line_of_position.py
│   │   │   ├── newton_raphson.py
│   │   │   ├── normalization.py
│   │   │   └── target_coordinates.py
│   │   ├── build_model.py # サンプルデータを元に機械学習のモデルを作成するプログラム
│   │   ├── collect_evaluation_data.py # 検証データを収集し，evaluation_dataに保存するプログラム
│   │   ├── collect_fingerprint.py # たぶん今後は使わない
│   │   ├── collect_sample_data.py # 機械学習のモデルを作成するためのサンプルデータを収集するプログラム
│   │   ├── evaluate_model.py # 機械学習のモデルの性能を評価するプログラム
│   │   ├── feature # 特徴量を算出するプログラム（詳細は各プログラム参照）
│   │   │   ├── convex_hull_volume.py
│   │   │   ├── distance_error_squared.py
│   │   │   ├── distance_from_center_of_field_to_target.py
│   │   │   ├── distance_from_centroid_of_sn_available_to_tn_estimated.py
│   │   │   ├── distance_from_sensors_to_approximate_line.py
│   │   │   ├── distance_squared_from_sensors_linear_regression_to_target.py
│   │   │   ├── feature_extraction.py
│   │   │   ├── mse_sensors_linear_regression.py
│   │   │   └── residual_avg.py
│   │   ├── localize.py # 測位シミュレーションを行うコアプログラム
│   │   └── result # 測位結果を算出するためのプログラム群
│   │       ├── localizable_probability_distribution.py
│   │       └── rmse_distribution.py
│   ├── launch.py # functionsフォルダの最上層にあるプログラムがまとめて実行されるコアプログラム
│   ├── models # build_model.pyで作成したモデルがここに保存される
│   │   ├── collective
│   │   │   ├── lgb
│   │   │   ├── nn
│   │   │   ├── rf
│   │   │   └── svm
│   │   └── successive
│   │       ├── lgb
│   │       ├── nn
│   │       ├── rf
│   │       ├── svm
│   │       └── xgb
│   ├── models_example # 必要に応じて追加したいモデルをここへ（必要に応じてモデルのスコアをmodel_score.yamlにメモする）
│   │   ├── collective
│   │   │   ├── lgb
│   │   │   ├── nn
│   │   │   ├── rf
│   │   │   ├── svm
│   │   │   └── xgb
│   │   └── successive
│   │       ├── lgb
│   │       ├── nn
│   │       ├── rf
│   │       ├── svm
│   │       └── xgb
│   ├── output # 結果が実行した日時で出力される（そのときのconfigもバックアップされる）
│   ├── sample_data # 機械学習のモデルを作成するときに使用するサンプルデータ
│   │   ├── collective # 清水手法だと有意な特徴量が得られないのでサンプルとしては使えない（後述）
│   │   └── successive # 山本手法では特徴量が出やすいのでサンプルとして使える（後述）
│   └── sample_data_example # 必要に応じて追加したいサンプルデータをここへ
│       ├── collective
│       └── successive
└── README.md # このファイル
```
> [!Tips]
> functions最上層にあるプログラムは基本的にそれ単体で実行できるようになっています．複数プログラムを同時に実行したい場合はlaunch.pyを使用すれば良いですが，それだと時間がかかる場合があるので必要に応じて単体で実行してください．

## プログラムの実行方法

```bash
# 標準出力と標準エラー出力をターミナルに表示しながらバックグラウンドで実行
nohup python launch.py config_0 &

# 標準出力と標準エラー出力を無視してバックグラウンドで実行
nohup python launch.py config_0 > /dev/null 2>&1 &

# 標準出力を無視し、標準エラー出力をerror.logに記録してバックグラウンドで実行
nohup python launch.py config_0 > /dev/null 2> error.log &
```

## 提案協調測位アルゴリズムの注意点
1. ゼミ資料でも記載している通り，サンプルデータを取得する場合，清水手法では有意な特徴を得ることが難しい（凸包のパターンが少なくなるなど）ので，山本手法に従ってサンプルを取得した方がモデルの性能は向上する．
2. そのため，サンプルデータはsuccesive，検証データはcollectiveで収集する方が良い．
3. それでいうと，successiveで最適化しても，collectiveでは最適ではないという点に注意が必要．その逆も然り．システム的には再現率や成功率に限界がある．（改善の余地あり？）
4. 山本修論に従ってサンプルデータを取得し作成した機械学習のモデルは，サービスエリアの中心に存在するTNの測位誤差が大きいと予測されることがある．
5. これは事前学習で9つの候補点に設置したTNがサービスエリアの中心で推定されることがあり，このときの推定誤差は2mを大きく超えてしまうためである．
6. 清水手法では，取得した測距値を遠い位置に存在するAN・RNのものから順番に削除しながら再帰的に機械学習モデルに通し，そのなかで陰性と判断されるか，推定位置がサービスエリアの中心で収束するようになっている．
7. 特徴量のなかで重要なのは主に凸包と残差である．他のパラメータは補助的なものになっている．

## 今後の検討
外側のターゲットに関しては緩和法を用いると良いかも．
NLOSを考えたときに，残差が大きくなってしまうのでどうなるか気になる．基準信号送信回数が少ないと単純に測位されないTNが増えそうな気がする．

