import numpy as np

def value(sensors: np.ndarray, distances: np.ndarray) -> np.ndarray:
  sensors_x = sensors[:, 0]
  sensors_y = sensors[:, 1]

  sensors_count = len(sensors) - 1

  # R matrix
  R_matrix = 2*(sensors[sensors_count, :2] - sensors[:sensors_count, :2])

  # I matrix
  I_matrix = (distances[:sensors_count]**2 - sensors_x[:sensors_count]**2 - sensors_y[:sensors_count]**2) \
           - (distances[sensors_count]**2 - sensors_x[sensors_count]**2 - sensors_y[sensors_count]**2)
  
  if np.linalg.det(R_matrix.T@R_matrix) == 0:
    return np.array([np.nan, np.nan])
  else:
    return np.linalg.inv(R_matrix.T@R_matrix)@(R_matrix.T)@I_matrix # (R^T*R)^(-1)*I

# Example usage
# sensors = np.array([[10, 10, 1], [10, 20, 1], [20, 10, 1], [20, 20, 1]])
# distances = np.array([1.2, 0.2, 1.0, 0.3])
# print(value(sensors, distances))

  # # R 
  # R_matrix = np.zeros((len(sensors) - 1 , 2))
  # for i in range(len(sensors) - 1):
  #   R_matrix[i, 0] = 2*(sensors[-1, 0] - sensors[i][0])
  #   R_matrix[i, 1] = 2*(sensors[-1, 1] - sensors[i][1])
    # I
  # I_matrix = np.zeros((len(sensors) - 1 , 1))
  # for i in range(len(sensors) - 1):
  #   I_matrix[i] = -(sensors[i, 0])**2.0 - (sensors[i, 1])**2.0 + (sensors[i, 3])**2.0 - (-(sensors[-1, 0])**2.0 - (sensors[-1, 1])**2 + (sensors[-1, 3])**2.0)
  
  