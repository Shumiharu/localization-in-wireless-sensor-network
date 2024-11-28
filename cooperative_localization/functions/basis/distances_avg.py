import numpy as np

def calculate(distances_measured_list_avg: np.ndarray, distances_measured_list: np.ndarray, distances_measured_list_count: np.ndarray, mask_sensors_measured: np.ndarray):
    # print(distances_measured_list)
    # print(distances_measured_list_avg)

    # 測距が上限になったセンサの数がある場合に上限になったセンサの部分の値を保存,前回施行時の配列の初期値が今回のものと同じになるよう成型
    if len(mask_sensors_measured) > 0:
        tmp_distances_measured_list_avg = distances_measured_list_avg[mask_sensors_measured]
        tmp_distances_measured_list_count = distances_measured_list_count[mask_sensors_measured]
        distances_measured_list_avg = np.delete(distances_measured_list_avg, mask_sensors_measured, axis = 0)
        distances_measured_list_count = np.delete(distances_measured_list_count, mask_sensors_measured, axis = 0)

    sensors_count = distances_measured_list.shape[0] 
    sensors_count_previous = distances_measured_list_avg.shape[0]
    difference_sensors_count = sensors_count - sensors_count_previous
    
    # 前回の試行時よりもセンサー数が増えていたらセンサー不足分を0で補う
    if difference_sensors_count > 0:
        # ターゲット数の計算
        targets_count = distances_measured_list.shape[1]
        # 不足していたセンサー分の行×ターゲット数の列の０行列の作成
        zeros_for_completion = np.zeros((difference_sensors_count, targets_count))
        #　前文で作成していた行列をひとつ前の結果に垂直方向に接続
        distances_measured_list_avg = np.vstack((distances_measured_list_avg, zeros_for_completion))
        distances_measured_list_count = np.vstack((distances_measured_list_count, zeros_for_completion))

    if len(distances_measured_list) == 0:
        distances_measured_list = tmp_distances_measured_list_avg
        distances_measured_list_count = tmp_distances_measured_list_count
        return distances_measured_list, distances_measured_list_count
    
    # 測位できていないもののみ実行する
    mask_unlocalized = ~np.isnan(distances_measured_list).any(axis=0)
    for index_unlocalized in np.where(mask_unlocalized)[0]:

        distances_measured = distances_measured_list[:, index_unlocalized]
        distances_measured_avg = distances_measured_list_avg[:, index_unlocalized]

        distances_measured_count = np.where(np.isinf(distances_measured), 0, 1)
        
        distances_measured = np.where(distances_measured == -np.inf, 0, distances_measured)
        distances_measured_avg = np.where(distances_measured_avg == -np.inf, 0, distances_measured_avg)
        # print(f"distances_measured: {distances_measured}")
        # print(f"distances_measured_avg: {distances_measured_avg}")
        # print(f"distance_measured_list_count: {distances_measured_list_count[:, index_unlocalized]}")

        distances_measured_count_for_avg = distances_measured_list_count[:, index_unlocalized] + distances_measured_count
        distances_measured_count_for_avg[distances_measured_count_for_avg == 0] = 1

        distances_measured_list[:, index_unlocalized] = (distances_measured_avg*(distances_measured_list_count[:, index_unlocalized]) + distances_measured)/distances_measured_count_for_avg
        # print(f"distances_measured_list[:, index_unlocalized]: {distances_measured_list[:, index_unlocalized]}")
        distances_measured_list[:, index_unlocalized] = np.where(distances_measured_list[:, index_unlocalized] == 0., -np.inf, distances_measured_list[:, index_unlocalized])

    distances_measured_list_count += np.where(np.isinf(distances_measured_list), 0, 1)
    if len(mask_sensors_measured) > 0:
        distances_measured_list = np.vstack((tmp_distances_measured_list_avg, distances_measured_list))
        distances_measured_list_count = np.vstack((tmp_distances_measured_list_count, distances_measured_list_count))
    return distances_measured_list, distances_measured_list_count

# Example Usage
# distances_measured_count_for_avg = np.array([[3, 16, -np.inf],[-np.inf, -np.inf, 14], [17, -np.inf, 11], [20, -np.inf, 3]])
# b = np.array([[np.nan, 4, np.nan], [np.nan, -np.inf, np.nan], [np.nan, -np.inf, np.nan],[np.nan, -np.inf, np.nan], [np.nan, -np.inf, np.nan], [np.nan, 15, np.nan]])
# c = np.array([[1, 1, 0], [0, 0, 1],[1, 0, 1],[1, 0, 1]])
# print(calculate(distances_measured_count_for_avg, b, c))
