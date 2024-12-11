# モデルが予測した値から確率的に01を判断する部分は問題なさそう？
# 再帰部分が怪しい？
import numpy as np
import itertools
# import random
# for i in range(1000000):
#     ransuu = 5
#     ransuu_probable = ransuu/10
#     if ransuu_probable == 0:
#         change_ransuu_01 = random.choices([0,1],weights=[ransuu_probable,(1.0-ransuu_probable)],k=1)
    
#     else:
#         change_ransuu_01 = random.choices([0,1],weights=[0.5 + ransuu_probable,(0.5-ransuu_probable)],k=1)
#     if ransuu_probable == 0.5 and change_ransuu_01 == 1:
#         print("何か違う")
#     if ransuu_probable == 0 and change_ransuu_01 ==0:
#         print("何か違う")

# rollman = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
# print(f"{rollman}")
# mask_delete_number = np.array([0,2])
# tmp_rollman = rollman[mask_delete_number]
# rollman = np.delete(rollman,mask_delete_number,axis=0)
# print(f"\n after_delete:\n{rollman}")
# for delete_number, tmp_delete_data in zip(mask_delete_number,tmp_rollman):
#     rollman = np.insert(rollman,delete_number,tmp_delete_data,axis=0)
# print(f"\n recover_data:\n{rollman}")

# import numpy as np

# # 例として3x3のNumPy配列を作成
# array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# # 3行目（インデックス2）を0で満たす
# array[2] = 0

# print(array)


# max_and_index = np.array([1,2,3,4,5])
# max_number = np.max(max_and_index)
# index_number = np.where(max_and_index == max_number)
# print(f"{max_number} , {index_number}")
# print("終了")
# import numpy as np

combination_man = np.arange(5)
tansakun = itertools.combinations(combination_man,2)
for tansakuns in tansakun:
    print(tansakuns)

    