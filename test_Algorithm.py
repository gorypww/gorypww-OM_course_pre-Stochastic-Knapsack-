import random

def generate_test_cases():
    test_cases = []
    for _ in range(100):
        n = random.randint(1, 10)
        weights = [random.randint(1, 10) for _ in range(n)]
        values = [random.randint(1, 10) for _ in range(n)]
        capacity = random.randint(1, 20)
        test_cases.append((weights, values, capacity))
    return test_cases

def test_algorithm():
    test_cases = generate_test_cases()
    for i, (weights, values, capacity) in enumerate(test_cases):
        optimal_solution_dp = knapsack_problem(weights, values, capacity)
        optimal_solution_ip = solve_binaryIP(values, [weights], [capacity])
        if optimal_solution_dp != optimal_solution_ip:
            print(f"Test case {i+1}: Optimal solutions are different.")
            print(f"Weights: {weights}")
            print(f"Values: {values}")
            print(f"Capacity: {capacity}")
            print(f"Optimal solution (DP): {optimal_solution_dp}")
            print(f"Optimal solution (IP): {optimal_solution_ip}")
            print()
    print("Testing complete.")

test_algorithm()