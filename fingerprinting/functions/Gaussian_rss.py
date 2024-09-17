from typing import Dict
import math 
import random

def value(sigma: float, average_rss: float) -> float:
  u1: float = (random.uniform(0, 10**9)+1)/((10**9)+1)
  u2: float = (random.uniform(0, 10**9)+1)/((10**9)+1)
  sample_rss: float = sigma*math.sqrt(-2.0*math.log(u1))*math.cos(2.0*math.pi*u2) + average_rss
  return sample_rss