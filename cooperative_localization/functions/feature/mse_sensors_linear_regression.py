import numpy as np

def calculate(sensors_available: np.ndarray) -> float:
    sensors_x = sensors_available[:, 0]
    sensors_x += np.random.normal(0, 1e-10, size=sensors_x.shape)  # 近似直線算出のための微小なノイズの追加

    sensors_y = sensors_available[:, 1]
    sensors_y += np.random.normal(0, 1e-10, size=sensors_y.shape)  # 近似直線算出のための微小なノイズの追加

    # 線形回帰の係数を計算 y = ax + b, a = slope, b = intercept
    A = np.vstack([sensors_x, np.ones(len(sensors_x))]).T
    slope, intercept = np.linalg.lstsq(A, sensors_y, rcond=None)[0]

    # 予測値を計算
    sensors_predicted_y = slope * sensors_x + intercept

    # MSEを計算
    mse = np.mean((sensors_y - sensors_predicted_y) ** 2)

    return mse

# Example Usage
# sensors_available = np.array([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0], [10.0, 20.0, 1.0], [20.0, 20.0, 1.0]])
# print(f"distance_from_sensors_to_approximate_line = {calculate(sensors_available)}")


