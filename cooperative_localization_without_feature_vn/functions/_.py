import numpy as np

# 初期化する空の配列（行数は0、列数は3と仮定）
arr = np.empty((0, 3))

# 追加する配列
new_row = np.array([1, 2, 3])

# 行方向に追加する
arr = np.append(arr, [new_row], axis=0)

# さらに別の行を追加
another_row = np.array([4, 5, 6])
arr = np.append(arr, [another_row], axis=0)

print(arr)
