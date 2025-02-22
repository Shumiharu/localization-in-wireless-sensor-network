import numpy as np

# 近似直線とセンサーの距離を算出
def calculate(sensors_available: np.ndarray) -> float:
    sensors_x = sensors_available[:, 0]
    sensors_x += np.random.normal(0, 1e-10, size=sensors_x.shape)  # 近似直線算出のための微小なノイズの追加

    sensors_y = sensors_available[:, 1]
    sensors_y += np.random.normal(0, 1e-10, size=sensors_y.shape)  # 近似直線算出のための微小なノイズの追加

    sensors_available_count = len(sensors_available)

    # 線形回帰の係数を計算 y = ax + b, a = slope, b = intercept
    # slope = (np.sum(sensors_x*sensors_y) - np.sum(sensors_x)*np.sum(sensors_y)/sensors_available_count)/(np.sum(sensors_x**2) - np.sum(sensors_x)**2/sensors_available_count)
    # intercept = np.mean(sensors_y) - slope * np.mean(sensors_x)
    slope, intercept = np.polyfit(sensors_x, sensors_y, 1)

    # 距離の計算 |αx + βy + γ|/√(α^2 + β^2)
    distance = np.sum(np.abs(slope*sensors_x - 1*sensors_y + intercept)/np.sqrt(slope**2 + 1**2))

    return distance/sensors_available_count

# Example Usage
# sensors_available = np.array([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0], [10.0, 20.0, 1.0], [20.0, 20.0, 1.0]])
# print(f"distance_from_sensors_to_approximate_line = {calculate(sensors_available)}")


