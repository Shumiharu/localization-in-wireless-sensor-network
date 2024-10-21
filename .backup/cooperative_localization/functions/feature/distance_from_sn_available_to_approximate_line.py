import numpy as np

# 近似直線とセンサーの距離を算出
def calculate(sensors_available: np.ndarray) -> float:
    sensors_x = sensors_available[:, 0]
    sensors_y = sensors_available[:, 1]
    sensors_available_count = len(sensors_available)

    # 線形回帰の係数を計算 y = ax + b, a = slope, b = intercept
    slope, intercept = np.polyfit(sensors_x, sensors_y, 1) # Copilotによる提案

    # 距離の計算 |αx + βy + γ|/√(α^2 + β^2)
    distance = np.sum(np.abs(slope*sensors_x - 1*sensors_y + intercept)/np.sqrt(slope**2 + 1**2))

    return distance/sensors_available_count

# Example Usage
# sensors_available = np.array([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0], [10.0, 20.0, 1.0], [20.0, 20.0, 1.0]])
# print(f"distance_from_sensors_to_approximate_line = {calculate(sensors_available)}")


