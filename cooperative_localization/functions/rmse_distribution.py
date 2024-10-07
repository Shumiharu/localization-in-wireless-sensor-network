import numpy as np 

def update(field_rmse_distribution: np.ndarray, grid_interval: float, targets: np.ndarray, squared_error_list: np.ndarray):
  targets_rounded = np.round(targets, decimal_places(grid_interval))
  for target_rounded, squared_error in zip(targets_rounded, squared_error_list):
    indices: np.ndarray = np.where(np.all(np.isclose(field_rmse_distribution[:, :2], target_rounded[:2]), axis=1))[0] # これは難しい
    if indices.size > 0:
      for index in indices:
        data_count = field_rmse_distribution[index, 3]
        field_rmse_distribution[index, 2] = np.sqrt((field_rmse_distribution[index, 2]*data_count + squared_error)/(data_count + 1))
        field_rmse_distribution[index, 3] += 1
  return field_rmse_distribution

def decimal_places(number: float) -> int:
    decimal_str = str(number).split('.')
    return len(decimal_str) - 1

# Examle Usage
# grid_interval = 0.1
# x_range = np.arange(0.0, 30.0 + grid_interval, grid_interval)
# y_range = np.arange(0.0, 30.0 + grid_interval, grid_interval)
# field_rmse_distribution = np.array([[x, y, 0.0, 0] for x in x_range for y in y_range])
# targets = np.array([[0.1, 10.1, 1.0],[14.3, 11.7, 1.0],[17.7, 29.2, 1.0],[17.0, 15.6, 1.0]])
# squared_error_list = np.array([0.1, 0.803103358161658, 1.3459667737186787, 0.1121262089814448])
# field_rmse_distribution = update(field_rmse_distribution, grid_interval, targets, squared_error_list)
# for i, row in enumerate(field_rmse_distribution[:, 2]):
#    if not row == 0:
#       print(f"value = {row}, index = {i}")



