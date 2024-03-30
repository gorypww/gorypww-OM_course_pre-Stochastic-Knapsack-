import numpy as np
import Algorithm as alg
import math
import matplotlib.pyplot as plt
plt.ioff()

np.random.seed(0)

# 探究算法1返回的可行库存量Q，收敛的速度是怎样的
## 输入
N = 3 # 客户数量
max_gap = 0.1 # 求解CCP问题的gap
max_iteration = math.ceil(4 * N / max_gap**2) + 1 # 循环上限（算法收敛速度）
tolerance = 0.05 # 由于type I 总是达不到的，还需要给一个容忍度
print_info = False

## 生成数据
def draw_gap(alpha_list, Q, print_info, max_gap, max_iteration):
    N = len(alpha_list)
    r_array = np.array([0.000001]*N) # 初始化r(0)
    iteration = 1
    fill_rate_list = []
    gap_list = []
    temp_fill_rate = 0
    while temp_fill_rate < 1 and iteration < max_iteration:
        A = alg.generate_one_demand_uniform(N)
        A = A.reshape(1, -1)
        solution, _ = alg.solve_binaryIP(r_array, A, [Q])
        if solution is not None:
            r_array = [r_array[i] + alpha_list[i] - solution[i] for i in range(N)]
            temp_fill_rate = np.sum(np.array(r_array) < tolerance*iteration)/N
            temp_gap = math.sqrt(sum([(r/iteration)**2 for r in r_array if r > 0]))
        elif print_info:
            print("function calculate_temp_gap debug")
        iteration += 1
        fill_rate_list.append(temp_fill_rate)
        gap_list.append(temp_gap)
    return fill_rate_list, gap_list

## 画图
alpha_list = np.random.rand(N)
Q = alg.algorithm_one(alpha_list, max_gap, max_iteration, print_info)
fill_rate_list, gap_list = draw_gap(alpha_list, Q, print_info, max_gap, max_iteration)
plt.plot(range(len(fill_rate_list)), fill_rate_list)
plt.xlabel('Time')
plt.ylabel('% of achieving service rate customers')
plt.show()


