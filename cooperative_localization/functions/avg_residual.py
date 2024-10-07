import numpy as np

def calculate(sensors_available: np.ndarray, target_localized: np.ndarray) -> float:
    distances_estimated = sensors_available[:, 2]
    sensors_available_count = len(sensors_available)
    avg_residual = np.sum((distances_estimated - np.linalg.norm(sensors_available[:, :1] - target_localized))**2)/sensors_available_count
    return avg_residual

    # count = 0
    # temp = 0
    # for j in range(Num_sensor):
    #     if TOA_Measurement_value[j] != None_distance:
    #         xk = Sensor_location[j,0]-x
    #         yk = Sensor_location[j,1]-y
    #         temp += (TOA_Measurement_value[j]-math.sqrt(xk**2+yk**2))**2
    #         count += 1

    residuals_x = sensors_available[:, 0] - target_localized[0]
    residuals_y = sensors_available[:, 1] - target_localized[1]