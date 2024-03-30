import numpy as np
import math
import matplotlib.pyplot as plt
import Algorithm as alg
import copy
plt.ioff()

np.random.seed(0)

# 探究算法1返回的可行库存量Q，收敛的速度是怎样的
## 输入
N = 3 # 客户数量
max_gap = 0.1 # 求解CCP问题的gap
max_iteration = math.ceil(4 * N / max_gap**2) + 1 # 循环上限（算法收敛速度）
tolerance = 0.05 # 由于服务水平总是无法达到的，还需要给一个容忍度
print_info = False

## 生成数据
def draw_gap(alpha_list, Q, print_info, max_gap, max_iteration, tolerance, compete_flag=False):
    N = len(alpha_list)
    r_array = np.array([0.000001]*N)
    iteration = 1
    fill_rate_list = []
    gap_list = [0]
    temp_fill_rate = 0
    compete = [[] for _ in range(N)]
    while temp_fill_rate < 1 and iteration < max_iteration:
        A = alg.generate_one_demand_uniform(N)
        A = A.reshape(1, -1)
        solution, _ = alg.solve_binaryIP(r_array, A, [Q])
        if solution is not None:
            r_array = [r_array[i] + alpha_list[i] - solution[i] for i in range(N)]
            if compete_flag:
                for i in range(N):
                    compete[i].append(r_array[i]/iteration)
            temp_fill_rate = np.sum(np.array(r_array) < tolerance*iteration)/N
            temp_gap = (gap_list[-1]*(iteration-1)+sum(solution))/iteration
        elif print_info:
            print("function calculate_temp_gap debug")
        iteration += 1
        fill_rate_list.append(temp_fill_rate)
        gap_list.append(temp_gap)
    if compete_flag:
        return compete
    return fill_rate_list, gap_list

## 画图
alpha_list = np.random.rand(N)
Q = alg.algorithm_one(alpha_list, max_gap, max_iteration, print_info)
fill_rate_list, gap_list = draw_gap(alpha_list, Q, print_info, max_gap, max_iteration, tolerance)
plt.plot(range(len(fill_rate_list)), fill_rate_list)
plt.xlabel('Time')
plt.ylabel('% of achieving service rate customers')
plt.show()
plt.plot(range(len(gap_list)), gap_list)
plt.xlabel('Time')
plt.ylabel('temp~/target~')
plt.axhline(y=sum(alpha_list), color='red', linestyle='-', label='Target_fill_rate')
plt.show()
np.random.seed(12)


# 虚高服务水平是否是更有策略
## 输入
alpha_true = [0.6,0.6,0.6]
alpha_compete = copy.deepcopy(alpha_true)
alpha_compete[0] = 0.8
max_iteration = 100

## 生成数据
def compete(alpha_lie, alpha_true, Q, print_info, max_gap, max_iteration, compete_flag=False):
    N = len(alpha_lie)
    r_array = np.array([0.000001]*N)
    iteration = 1
    fill_rate_list = []
    gap_list = []
    temp_fill_rate = 0
    compete = [[0] for _ in range(N)]
    while iteration < max_iteration:
        A = alg.generate_one_demand_uniform(N)
        A = A.reshape(1, -1)
        solution, _ = alg.solve_binaryIP(r_array, A, [Q])
        if solution is not None:
            r_array = [r_array[i] + alpha_lie[i] - solution[i] for i in range(N)]
            if compete_flag:
                for i in range(N):
                    compete[i].append((compete[i][-1]*(iteration-1)+solution[i])/iteration)
            temp_fill_rate = np.sum(np.array(r_array) < 0)/N
            temp_gap = math.sqrt(sum([(r/iteration)**2 for r in r_array if r > 0]))
        elif print_info:
            print("function calculate_temp_gap debug")
        iteration += 1
        fill_rate_list.append(temp_fill_rate)
        gap_list.append(temp_gap)
    if compete_flag:
        return compete
    return fill_rate_list, gap_list


Q_true = alg.algorithm_one(alpha_true, max_gap, max_iteration, False)
Q_compete = alg.algorithm_one(alpha_compete, max_gap, max_iteration, False)
nocompete_converge = compete(alpha_true, alpha_true, Q_true, print_info, max_gap, max_iteration, compete_flag=True)
compete_converge = compete(alpha_compete, alpha_true, Q_compete, print_info, max_gap, max_iteration, compete_flag=True)

for i in range(len(nocompete_converge)):
    plt.plot(range(len(nocompete_converge[i])), nocompete_converge[i], label='No Competition', linestyle='-',color='blue')
    plt.plot(range(len(compete_converge[i])), compete_converge[i], label='Competition', linestyle='--',color = 'green')
    plt.axhline(y=alpha_true[i], color='red', linestyle='-', label='Target_fill_rate')
    plt.xlabel('Time')
    plt.ylabel('Temp_fill_rate')
    plt.legend()
    plt.show()

