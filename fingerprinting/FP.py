# フィンガープリントによる測位

#        4 m
# 4■---□---□---□---■
#  |               |
# 3□   ○   ○   ○   □
#  |               |
# 2□   ○   ●   ○   □ 4 m
#  |               |
# 1□   ○   ○   ○   □
#  |               |
# 0■---□---□---□---■
#  0   1   2   3   4

# Modeling of Indoor Positioning Systems Based on.pdf より
# □: The outer most positions reserved for placing access points only -> APを設置するための点
# ■: AP
# ○: The neighboring positions with location ﬁngerprints recorded in the database during the site-survey -> RSSがデータベースに保存されたフィンガープリントを持った点
# ●: mobile station -> 移動点TN(2, 2)

# プログラムの流れ
# 各パラメータ（sigma, path_loss_exponent）にしたがって
# 近傍点と移動点（●と○）の9点に対してそれぞれのAP（■または□）からRSSベクトル（フィンガープリント）を記録する(-> fingerprint_measurement)
# TN(2, 2)にガウス分布に従うRSSベクトルを生成し，(-> Gaussian_rss)それぞれのフィンガープリントとのユークリッド距離をとる．
# うち最小のものを推定点とみなして座標を出力し，(2, 2)であればtrue，それ以外ではfalseを返し(-> is_correct_location)記録する．

import os
import csv
from array import array
from functions import Gaussian_rss
from functions import fingerprint_measurement
from functions import is_correct_location
from functions import rss_measurement

if __name__ == "__main__":
  # ファイルの初期化
  file: str = os.path.basename(__file__)
  file_name: str = os.path.splitext(file)[0]
  output_file: str = "./output/" + file_name + ".csv"
  output_file_name: str = file_name + ".csv"
  
  with open(output_file, "w") as f:
    pass
  print(output_file_name + " was initialized.")

  print("method: Finger Printing")

  # 領域の1辺の長さ
  width: float = 4.0
  height: float = 4.0

  # 座標定義
  x_top: float = 4.0
  y_top: float = 4.0
  x_bottom: float = 0.0
  y_bottom: float = 0.0

  # 測定範囲
  mesured_range : dict = {
    "x_top": width,
    "x_bottom": 0.0,
    "y_top": height,
    "y_bottom": 0.0
  }
  print("field: " + str(width) + " m x " + str(height) + " m")

  # accuracy -> 精度，グリッドの間隔
  interval: float = 1.0
  rows: int = height/interval
  columns: int = width/interval
  print("interval: " + str(interval) + " m")

  # LOS or NLOS
  is_nlos: int = 1
  print("environment: NLOS" if is_nlos == 1 else "environment: LOS")

  if is_nlos:
    path_loss_per_meter: float = 37.7 # 自由空間経路損失
    path_loss_exponent: float = 3.3 # 経路損失指数
  else:
    path_loss_per_meter: float = 41.5
    path_loss_exponent: float = 2.0

  # default AP
  default_access_point = [
    {"x": x_bottom, "y": y_bottom}, #1
    {"x": x_top, "y": y_top}, #2
    {"x": x_bottom, "y": y_top}, #3
    {"x": x_top, "y": y_bottom}, #4
    {"x": 2.0, "y": 0.0}, #5
    {"x": 4.0, "y": 2.0}, #6
    {"x": 2.0, "y": 4.0}, #7
    {"x": 0.0, "y": 2.0}, #8
    {"x": 1.0, "y": 0.0}, #9
    {"x": 4.0, "y": 1.0}, #10
    {"x": 3.0, "y": 4.0}, #11
    {"x": 0.0, "y": 3.0}, #12
    {"x": 3.0, "y": 0.0}, #13
    {"x": 4.0, "y": 3.0}, #14
    {"x": 1.0, "y": 4.0}, #15
    {"x": 0.0, "y": 1.0}, #16
  ]

  # 有効なAP数
  valid_access_point_count: int = 4
  print("valid AP count: " + str(valid_access_point_count))

  # AP
  access_point: array = []
  is_adjacent: int = 1 # 対角か隣接か
  if is_adjacent:
    default_access_point[1], default_access_point[2] = default_access_point[2], default_access_point[1]  # 隣接なら入れ替え

  for i in range(valid_access_point_count):
    access_point.append(default_access_point[i])
  print("AP layout: adjacent" if is_adjacent else "AP layout: diagonal" )
  print(access_point)
  
  # TNの座標
  target = {"x": 2.0, "y": 2.0}
  print("target: (x, y) = (" + str(target["x"]) + ", " + str(target["y"]) + ")")

  # パラメータの設定
  is_default_sigma: int = 0
  if is_default_sigma :
    is_default_path_loss_exponent: int = 0
    max_sigma: int = 2 # for文を1回で終わらせるために設定
    double_max_path_loss_exponent: float = 13.0
  else:
    is_default_path_loss_exponent: int = 1
    max_sigma: float = 21.0
    double_max_path_loss_exponent: int = 2 # for文を1回で終わらせるために設定
  print("parameter: path loss exponent" if is_default_sigma else "parameter: sigma")

  # 送信電力
  transmit_power = 15.0

  # 標準偏差が固定の場合
  default_sigma: float = 2.13

  # パスロス指数が固定の場合
  default_path_loss_exponent = path_loss_exponent

  # 試行回数
  max_cycle: int = 10000
  print("cycle: " + str(max_cycle) + "\n")
  
  for sigma in range(1, int(max_sigma)):
    for double_path_loss_exponent in range(1, int(double_max_path_loss_exponent)):
      # パラメータ切り替え
      if is_default_sigma: 
        sigma = default_sigma
        path_loss_exponent = double_path_loss_exponent/2 # 0.5刻みのため
      if is_default_path_loss_exponent: path_loss_exponent = default_path_loss_exponent
    
      # 各パラメータの値ごとの進捗
      progress: int = 0

      # 正しく位置推定できた回数
      correct: int = 0

      # fingerprintの測定
      fingerprint: array = fingerprint_measurement.value(access_point, rows, columns, interval, transmit_power, path_loss_per_meter, path_loss_exponent)
      
      for cycle in range(max_cycle): # max_cycleだけ
        rss: array = []
        average_rss = rss_measurement.value(access_point, target, transmit_power, path_loss_per_meter, path_loss_exponent) # TNの平均RSSを測定する
        # 今のプログラムは1回しかRSSを測定していない．サンプルRSSを複数測定し．平均をとる考え方だとこのプログラムを書き換えないといけない．
        for i in range(valid_access_point_count): # 有効なAPから
          rss.append(Gaussian_rss.value(sigma, average_rss[i])) # サンプルRSSを生成し
          # rss.append(average_rss[i]) # 確認用（確率は常に1）
        is_correct: float = is_correct_location.value(rss, fingerprint, target) # 正しい座標点を推定できているかを判定する
        if is_correct: correct += 1 # 正しく推定されていれば推定成功数を+1する
        progress += 1
        print("\r" + "{:.2f}".format(progress/(max_cycle)*100) + "%" + " done.", end="")
      probability_of_correct_location = correct/(max_cycle) # 確率を算出する
      print(" sigma = " + str(sigma) + ", path loss exponent = " + str(path_loss_exponent) + ", Probability of Correct Location = " + "{:.3f}".format(correct/progress))

      output = [sigma, path_loss_exponent, probability_of_correct_location] 
      with open(output_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow(output)
  print("\n")
  print("complete.")
    

# メモ
# RSSベクトルは2種類存在する
# 1. エリア内のN個のアクセスポイントから移動局で測定されたRSSのサンプル（ランダム生成）
# 2. N個のアクセスポイントから特定の位置におけるすべての受信信号強度確率変数の真の平均値
# 1.は2を中心にガウス分布している
# ユークリッド距離が最も小さいものがその地点における座標であると推定される

