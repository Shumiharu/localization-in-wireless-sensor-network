import numpy as np

array1 = np.array([[10, 10,  1], [10, 20, 1], [11.5, 10.2, 1], [12.3, 28.9, 1], [21.2, 18.1, 1]])
array2 = np.array([[10, 10,  1], [10, 20, 1], [20, 10, 1], [20, 20, 1]])
array3 = np.array([1, 2, 3, 4, 5])  # array3のサイズをarray1に合わせる

# array1の各行がarray2に存在するかどうかを確認
not_in_array2 = np.array([not any(np.all(row == array2, axis=1)) for row in array1])

# 存在しない行のインデックスを取得
not_in_array2_indices = np.where(not_in_array2)[0]

# そのインデックスに対応するarray3の値を取得
array3_values = array3[not_in_array2_indices]

# 最大値のインデックスを取得
max_value_index = not_in_array2_indices[np.argmax(array3_values)]

print("array1の中でarray2にないインデックスのarray3から最大値のインデックス:", max_value_index)
