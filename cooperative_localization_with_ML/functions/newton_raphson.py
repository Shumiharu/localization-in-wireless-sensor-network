import numpy as np

def value(sensors: np.ndarray, distances: np.ndarray, target: np.ndarray, max_loop: int, threshold: float) -> np.ndarray:
  radius_maximized = 0.5
  radius_step = 0.1
  radiuses = np.arange(0.0, radius_maximized + radius_step, radius_step)

  angle_step = 0.1
  angles = np.arange(0, 1, angle_step)

  for radius in radiuses:
    for angle in angles:
      target_candidate = np.array([target[0] + radius*np.cos(2*np.pi*angle), target[1] + radius*np.sin(2*np.pi*angle)])
      for loop in range(max_loop):
        Q_matrix = np.zeros((2,2))
        f, g = 0.0, 0.0
        for sensor, distance in zip(sensors, distances):
          vector_sensor_to_target = target_candidate - sensor[:2]
          a = vector_sensor_to_target[0]
          b = vector_sensor_to_target[1]
          distance_from_sensor_to_target_candidate = np.linalg.norm(vector_sensor_to_target)

          f += 2*(1 - distance/distance_from_sensor_to_target_candidate)*vector_sensor_to_target[0]
          g += 2*(1 - distance/distance_from_sensor_to_target_candidate)*vector_sensor_to_target[1]
          
          Q_matrix[0, 0] += 2*(distance/(distance_from_sensor_to_target_candidate**3))*(vector_sensor_to_target[0]**2) + 2*(1 - distance/distance_from_sensor_to_target_candidate)
          Q_matrix[0, 1] += 2*(distance/(distance_from_sensor_to_target_candidate**3))*vector_sensor_to_target[0]*vector_sensor_to_target[1]
          Q_matrix[1, 0] += 2*(distance/(distance_from_sensor_to_target_candidate**3))*vector_sensor_to_target[0]*vector_sensor_to_target[1]
          Q_matrix[1, 1] += 2*(distance/(distance_from_sensor_to_target_candidate**3))*(vector_sensor_to_target[1]**2) + 2*(1 - distance/distance_from_sensor_to_target_candidate)

          if np.linalg.det(Q_matrix) == 0:
            continue
        d = np.array([f, g])
        t = target_candidate - np.linalg.inv(Q_matrix)@np.array([f, g])
        variation = np.linalg.inv(Q_matrix)@np.array([f, g])
        if np.all(np.abs(variation) < threshold):
          target = target_candidate - variation
          break
        target_candidate -= variation
      else:
        continue
      break
    else:
      continue
    break
  return target

# Example usage
sensors = np.array([[10, 10, 1], [10, 20, 1], [20, 10, 1], [20, 20, 1]])
distances = np.array([4.28477379, 8.08750981, 8.4018349, 11.40869214])
target = np.array([12.23027697, 11.97112623])
max_loop = 10
threshold = 1e-08
print(value(sensors, distances, target, max_loop, threshold))



  # for radius_step_count in range(max_radius_step_count + 1):
  #   for angle_step_count in range(max_angle_step_count):
  #     candidate_coordinate: dict = {"x": target["x"] + radius_step*radius_step_count*math.cos(2*math.pi*angle_step*angle_step_count), "y": target["y"] + radius_step*radius_step_count*math.sin(2*math.pi*angle_step*angle_step_count)}
  #     # Newton Raphsonの処理
  #     for nr in range(max_loop):
  #       Q_matrix = np.zeros((2,2)) 

  #       f: float = 0.0 
  #       g: float = 0.0
        
  #       # 各センサについて
  #       for i in range(len(sensors)):
  #         diff_x = candidate_coordinate["x"] - sensors[i]["x"]
  #         diff_y = candidate_coordinate["y"] - sensors[i]["y"]
  #         d: float = (diff_x)**2 + (diff_y)**2 # 距離差
  #         f += (float)(2*(1-distances[i]/math.sqrt(d))*np.squeeze(diff_x)) # 15のx
  #         g += (float)(2*(1-distances[i]/math.sqrt(d))*np.squeeze(diff_y)) # 15のy
  #         Q_matrix[0, 0] += (float)(2*(distances[i]/(d**1.5))*(np.squeeze(diff_x))**2 + 2*(1-distances[i]/math.sqrt(d))) # fをxで偏微分
  #         Q_matrix[0, 1] += (float)(2*(distances[i]/(d**1.5))*np.squeeze(diff_x)*np.squeeze(diff_y)) # fをyで偏微分
  #         Q_matrix[1, 0] += (float)(2*(distances[i]/(d**1.5))*np.squeeze(diff_x)*np.squeeze(diff_y)) # gをxで偏微分
  #         Q_matrix[1, 1] += (float)(2*(distances[i]/(d**1.5))*(np.squeeze(diff_y))**2 + 2*(1-distances[i]/np.sqrt(d))) # gをyで偏微分
  #       differentiated_function: array = [f, g] # x,yで偏微分された関数
  #       temporary_candidate_coordinate = coordinate_as_list.value(candidate_coordinate) - np.linalg.inv(Q_matrix)@differentiated_function
  #       # 距離が閾値より小さいか
  #       index_x = abs((temporary_candidate_coordinate - coordinate_as_list.value(candidate_coordinate))[0])
  #       index_y = abs((temporary_candidate_coordinate - coordinate_as_list.value(candidate_coordinate))[1])
  #       if(index_x < threshold and index_y < threshold):
  #         target = coordinate_as_dict.value(temporary_candidate_coordinate)
  #         break
  #       candidate_coordinate = coordinate_as_dict.value(temporary_candidate_coordinate)
  #     else:
  #       continue
  #     break
  #   else:
  #     continue
  #   break
  # return target