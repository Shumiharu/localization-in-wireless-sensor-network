import numpy as np

def calculate(sensors: np.ndarray, target: np.ndarray):
    sensors_vanish_flag = np.delete(sensors,2,axis=1)
    target_vanish_flag = np.delete(target,2)
    distance = np.linalg.norm(sensors_vanish_flag - target_vanish_flag, axis=1)
    return distance

