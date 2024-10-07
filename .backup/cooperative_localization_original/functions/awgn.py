from typing import Dict
import math 
import random

def value(sigma: float) -> float:
  u1: float = (random.uniform(0, 10**9)+1)/((10**9)+1) # 桁数が多いほど細かく定義できるよね
  u2: float = (random.uniform(0, 10**9)+1)/((10**9)+1)
  noise_re: float = sigma*math.sqrt(-2.0*math.log(u1))*math.cos(2.0*math.pi*u2) 
  # noise_im: float = math.sqrt(-2.0*math.log(u1))*math.sin(2.0*math.pi*u2) # 基礎ゼミでは虚軸を考えたけど考える必要なし
  return noise_re