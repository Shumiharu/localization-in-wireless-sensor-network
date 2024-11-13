import numpy as np

def calculate(sensors_available: np.ndarray, target_estimated: np.ndarray) -> float:
    sensors_x = sensors_available[:, 0].astype(float)
    sensors_x += np.random.normal(0, 1e-10, size=sensors_x.shape)  # 近似直線算出のための微小なノイズの追加

    sensors_y = sensors_available[:, 1].astype(float)
    sensors_y += np.random.normal(0, 1e-10, size=sensors_y.shape)  # 近似直線算出のための微小なノイズの追加

    # 線形回帰の係数を計算 y = ax + b, a = slope, b = intercept
    slope, intercept = np.polyfit(sensors_x, sensors_y, 1)
    print(f"linear regression: y = {slope}x + {intercept}")

    # target_estimated座標点との距離の二乗を計算
    target_estimated_x = target_estimated[0]
    target_estimated_y = target_estimated[1]
    
    distance_squared = np.square(np.abs(slope*target_estimated_x - 1*target_estimated_y + intercept)/np.sqrt(slope**2 + 1**2))
    
    return distance_squared

# Example Usage
# sensors_available = np.array([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0], [10.0, 20.0, 1.0], [20.0, 20.0, 1.0]])
# target_estimated = np.array([0.0, 0.0, 0.0])
# print(f"mse_from_target_to_sensors_linear_regression = {calculate(sensors_available, target_estimated)}")