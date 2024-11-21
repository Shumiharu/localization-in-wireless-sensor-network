# モデルが予測した値から確率的に01を判断する部分は問題なさそう？
# 再帰部分が怪しい？

import random
for i in range(1000000):
    ransuu = 5
    ransuu_probable = ransuu/10
    if ransuu_probable == 0:
        change_ransuu_01 = random.choices([0,1],weights=[ransuu_probable,(1.0-ransuu_probable)],k=1)
    
    else:
        change_ransuu_01 = random.choices([0,1],weights=[0.5 + ransuu_probable,(0.5-ransuu_probable)],k=1)
    if ransuu_probable == 0.5 and change_ransuu_01 == 1:
        print("何か違う")
    if ransuu_probable == 0 and change_ransuu_01 ==0:
        print("何か違う")

print("終了")
    