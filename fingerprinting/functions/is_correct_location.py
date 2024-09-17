import math
from array import array

def value(rss: array, fingerprint: array, correct_location: dict):
  euclidean_distances: array = []
  for i in range(len(fingerprint)):
    squared_euclidean_distance: float = 0.0
    for j in range(len(rss)):
      squared_euclidean_distance += (fingerprint[i]["rss"][j] - rss[j])**2 # fingerprint[i]の座標におけるAP[j]番のfingerprint[i]のRSSとRSSの差の二乗
    euclidean_distances.append(math.sqrt(squared_euclidean_distance))
  
  estimate_location: array = []
  min_euclidean_distance_index: array = [euclidean_distances.index(min(euclidean_distances))] # 重複を許さない場合
  # min_euclidean_distance_index: array = [i for i, x in enumerate(euclidean_distances) if x == min(euclidean_distances)] # 重複を許す場合（デバッグ用）
  for i in range(len(min_euclidean_distance_index)):
    estimate_location.append({"x": fingerprint[min_euclidean_distance_index[i]]["x"], "y": fingerprint[min_euclidean_distance_index[i]]["y"]})
  for i in range(len(min_euclidean_distance_index)):
    if estimate_location[i]["x"] == correct_location["x"] and estimate_location[i]["y"] == correct_location["y"]: 
      return 1
  else:
    return 0


  


  
  