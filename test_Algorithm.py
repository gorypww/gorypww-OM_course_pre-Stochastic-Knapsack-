from Algorithm import *

# 测试
## 测试函数
def test_algorithm_one(N, max_gap, max_iteration):
    alpha_list = np.random.rand(N)
    Q = algorithm_one(alpha_list, max_gap, max_iteration, print_info=True)
    return Q

def test_algorithm_two(N, max_gap, max_iteration):
    alpha_list = np.random.rand(N)
    Q = generate_initial_bound_uniform(alpha_list)[1]*np.random.rand()
    print(Q)
    return algorithm_two(alpha_list, Q, max_gap, max_iteration, print_info=True)


## 测试实例
print(test_algorithm_one(10, 0.1, 100))
test_knapsack_problem()
print(test_algorithm_two(10, 0.1, 100))