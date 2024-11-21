import ast
import inspect
import numpy as np

from feature import convex_hull_volume
from feature import distance_from_center_of_field_to_target
from feature import distance_from_sensors_to_approximate_line
from feature import distance_squared_from_sensors_linear_regression_to_target
from feature import mse_sensors_linear_regression
from feature import residual_avg

def calculate(sensors_available, distances_estimated, target_estimated, field_range):

    return np.array([
        convex_hull_volume.calculate(sensors_available),
        distance_from_center_of_field_to_target.calculate(field_range, target_estimated),
        distance_from_sensors_to_approximate_line.calculate(sensors_available),
        # distance_squared_from_sensors_linear_regression_to_target.calculate(sensors_available, target_estimated),
        # mse_sensors_linear_regression.calculate(sensors_available),
        residual_avg.calculate(sensors_available, distances_estimated, target_estimated)
    ])

# 特徴量の数を返す
def count() -> int:
    sensors_available =  np.array([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0], [10.0, 20.0, 1.0], [20.0, 20.0, 1.0]])
    distances_estimated = np.array([7.071, 7.071, 7.071, 7.071])
    target_estimated = np.array([15., 15., 0.0])
    field_range = {
        "x_top": 30.0,
        "x_bottom": 0.0,
        "y_top": 30.0,
        "y_bottom": 0.0
    }
    features_count = len(calculate(sensors_available, distances_estimated, target_estimated, field_range)) + 1
    return features_count

# 現在有効な特徴量の名前をリストで取得
def get_features_name() -> list:
    feature_names = []
    code_calculate = inspect.getsource(calculate)
    tree = ast.parse(code_calculate)
    for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id != 'np':
                    feature_names.append(node.value.id)
    return list(feature_names)

