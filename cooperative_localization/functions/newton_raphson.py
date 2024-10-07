import numpy as np

def calculate(sensors: np.ndarray, distances: np.ndarray, target: np.ndarray, max_loop: int, threshold: float) -> np.ndarray:
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
# sensors = np.array([[10, 10, 1], [10, 20, 1], [20, 10, 1], [20, 20, 1]])
# distances = np.array([4.28477379, 8.08750981, 8.4018349, 11.40869214])
# target = np.array([12.23027697, 11.97112623])
# max_loop = 10
# threshold = 1e-08
# print(calculate(sensors, distances, target, max_loop, threshold))