import numpy as np
from functions import distance_toa

def calculate(sensors_available: np.ndarray, target_localized: np.ndarray, channel: dict, max_distance_measurement: int) -> float:
    distance_from_sensor_to_target_localized = np.linalg.norm(sensors_available[:, :2] - target_localized[:2])
    distance_estimated = distance_toa.calculate(channel, max_distance_measurement, distance_from_sensor_to_target_localized)

    centroid = np.array([np.mean(sensors_available[:, 0]), np.mean(sensors_available[:, 1])])
    distance_from_centroid_of_sensors_to_vn_maximized = 0
    if not np.isinf(distance_estimated):
        distance_from_centroid_of_sensors_to_vn_maximized = np.max(np.sqrt((centroid[0] - sensors_available[:, 0])**2 + (centroid[1] - sensors_available[:, 1])**2))

    return distance_from_centroid_of_sensors_to_vn_maximized
    

# def calculate(F_m):
#     temp_G = 0      #VNの最大距離
#     PL_0 = 41.33    #基本のパスロス電力
#     alpha = 1.96    #パスロス指数
#     PT = 15         #送信電力
#     sigma = (10**(0.1*2.73))**2 #シャドウイング考慮の分散
#     RSS_threshold = -48         #受信感度
#     G = np.array([0.0,0.0])     #SNの重心
#     count = 0
#     """SNの重心計算"""
#     for j in range(Num_sensor):
#         if TOA_Measurement_value[j] != None_distance:
#             G[0] += Sensor_location[j,0]
#             G[1] += Sensor_location[j,1]
#             count += 1
#     G[0] = G[0]/count
#     G[1] = G[1]/count
#     """VNの最大距離計算"""
#     for i in range(Num_sensor):
#         if TOA_Measurement_value[i] == None_distance:
#             xk = Sensor_location[i,0]-F_m[0]
#             yk = Sensor_location[i,1]-F_m[1]
#             l = math.sqrt(xk**2+yk**2)
#             noise = AWGN(sigma)
#             if noise >= 0:
#                 PL = PL_0 + 10*alpha*math.log10(l) + math.log10(noise)*10
#             else:
#                 PL = PL_0 + 10*alpha*math.log10(l) - math.log10(noise*(-1))*10
#             temp_sumple_RSS = PT - PL
#             G_dis = math.sqrt((G[0]-Sensor_location[i,0])**2+(G[1]-Sensor_location[i,1])**2)
#             if (temp_sumple_RSS >= RSS_threshold) and (G_dis >= temp_G):
#                 temp_G = G_dis
#     return temp_G