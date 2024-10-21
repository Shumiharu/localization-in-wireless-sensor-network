import numpy as np 

def update(field_distribution: np.ndarray, grid_interval: float, targets: np.ndarray, targets_localized: np.ndarray):
  for target in targets:
    # is_localized = np.any(np.all(targets_localized == target, axis=1))
    is_localized = target[2]

    target_rounded = np.round(target, decimal_places(grid_interval))
    indices = np.where(np.all(np.isclose(field_distribution[:, :2], target_rounded[:2]), axis=1))[0]
    for index in indices:
      data_count = field_distribution[index, 3]
      field_distribution[index, 2] = (field_distribution[index, 2]*data_count + is_localized)/(data_count + 1)
      field_distribution[index, 3] += 1
  return field_distribution

def decimal_places(number: float) -> int:
    decimal_str = str(number).split('.')
    return len(decimal_str[-1]) if '.' in str(number) else 0

# Examle Usage
# grid_interval = 0.1
# x_range = np.arange(0.0, 30.0 + grid_interval, grid_interval)
# y_range = np.arange(0.0, 30.0 + grid_interval, grid_interval)
# field_distribution = np.array([[x, y, 0.0, 0] for x in x_range for y in y_range])
# targets = np.array([[0.11, 10.12, 1], [2.34, 12.12, 1], [15.65, 1.14, 1], [0.43, 20.10, 1]])
# targets_localized = np.array([[0.11, 10.12, 1], [2.34, 12.12, 1], [15.65, 1.14, 1]])
# field_distribution = update(field_distribution, grid_interval, targets, targets_localized)
# targets = np.array([[0.11, 10.12, 1], [2.34, 12.12, 1], [15.65, 1.14, 1], [0.43, 20.10, 1]])
# targets_localized = np.array([[0.11, 10.12, 1], [2.34, 12.12, 1], [15.65, 1.14, 1], [0.43, 20.10, 1]])
# field_distribution = update(field_distribution, grid_interval, targets, targets_localized)
# print("probability")
# for i, row in enumerate(field_distribution[:, 2]):
#    if not row == 0:
#       print(f"value = {row}, index = {i}")
# print("data_count")
# for i, row in enumerate(field_distribution[:, 3]):
#    if not row == 0:
#       print(f"value = {row}, index = {i}")