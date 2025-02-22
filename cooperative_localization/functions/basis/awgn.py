import math 
import random

# ランダムシードの設定
# random.seed(42)

def calculate(standard_deviation: float) -> float:
  u1: float = (random.uniform(0, 10**9)+1)/((10**9)+1) # 桁数が多いほど細かく定義できるよね
  u2: float = (random.uniform(0, 10**9)+1)/((10**9)+1)
  noise_re: float = standard_deviation*math.sqrt(-2.0*math.log(u1))*math.cos(2.0*math.pi*u2) 
  # noise_im: float = math.sqrt(-2.0*math.log(u1))*math.sin(2.0*math.pi*u2) # 基礎ゼミでは虚軸を考えたけど考える必要なし
  return noise_re