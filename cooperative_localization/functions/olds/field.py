# 範囲を補正
def value(range: dict, coordinate: dict) -> dict:
  if(coordinate["x"] < range["x_bottom"]):
    coordinate["x"] = range["x_bottom"]
  elif(coordinate["x"] > range["x_top"]):
    coordinate["x"] = range["x_top"]
  if(coordinate["y"] < range["y_bottom"]):
    coordinate["y"] = range["y_bottom"]
  elif(coordinate["y"] > range["y_top"]):
    coordinate["y"] = range["y_top"]
  return coordinate