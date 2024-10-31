import numpy as np

from feature import convex_hull_volume
from feature import distance_from_center_of_field_to_target
from feature import distance_from_sensors_to_approximate_line
from feature import residual_avg

def calculate(sensors_available, distances_estimated, target_estimated, field_range):

    feature_convex_hull_volume = convex_hull_volume.calculate(sensors_available)
    feature_distance_from_center_of_field_to_target = distance_from_center_of_field_to_target.calculate(field_range, target_estimated)
    feature_distance_from_sensors_to_approximate_line = distance_from_sensors_to_approximate_line.calculate(sensors_available)
    feature_residual_avg = residual_avg.calculate(sensors_available, distances_estimated, target_estimated)
    
    return np.array([
        feature_convex_hull_volume,
        feature_distance_from_center_of_field_to_target,
        feature_distance_from_sensors_to_approximate_line,
        feature_residual_avg,
    ])