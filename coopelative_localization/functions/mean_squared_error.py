# MSEを求める
def value(target: dict, coordinate: dict, max_loop: int) -> float:
  return ((target["x"] - coordinate["x"])**2 + (target["y"] - coordinate["y"])**2)/max_loop # 誤差が大きいほどMSEは大きくなる
