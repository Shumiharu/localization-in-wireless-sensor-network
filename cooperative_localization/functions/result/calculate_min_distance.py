# 各センサにおける他センサとの距離の最小値を探索する
import numpy as np

def calculate(sensors:np.ndarray):
    min_distances = np.array([])
    vanish_localize_flag = np.delete(sensors,2,axis=1)

    for sensor in range(len(sensors)):
        distance_sensors = np.linalg.norm(vanish_localize_flag[sensor]-vanish_localize_flag,axis=1)
        distance_sensors[sensor] = np.inf

        nearest_distance = np.min(distance_sensors)

        min_distances=np.append(min_distances,nearest_distance)
    
    return min_distances
