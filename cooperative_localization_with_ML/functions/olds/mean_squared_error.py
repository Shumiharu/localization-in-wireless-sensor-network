# MSEを求める
def value(target: dict, target_localized: dict, max_loop: int) -> float:
  return ((target["x"] - target_localized["x"])**2 + (target["y"] - target_localized["y"])**2)/max_loop # 誤差が大きいほどMSEは大きくなる
