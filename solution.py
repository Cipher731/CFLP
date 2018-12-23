import numpy as np


class Solution:
    def __init__(self, sol_mat: np.ndarray, f_cap: np.ndarray, c_demand: np.ndarray, f_remain: np.ndarray):
        # All of these objects are past by reference, thus no performance issue.
        self._c_num = sol_mat.shape[0]  # customer number
        self._f_num = sol_mat.shape[1]  # facility number
        self._f_cap = f_cap  # Facility capacity, though not used.
        self._f_remain = f_remain  # facility remaining capacity
        self._c_demand = c_demand  # customer demand
        self._mat = sol_mat  # solution matrix
        self._cost = None  # cost cache

    def cost(self, f_cost: np.ndarray, c_cost: np.ndarray) -> int:
        if self._cost is not None:
            return self._cost
        f_open = self._mat.sum(axis=0, dtype=bool)
        f_total = f_cost.dot(f_open.T)
        c_total = c_cost[self._mat].sum()
        self._cost = f_total + c_total
        return self._cost

    def all_neighbors(self):
        # The neighbors are produced by reassigning one customer to a new capable facility.
        # Thus the number of total neighbors is exactly the customer number.
        neighbors = []
        for c in range(self._c_num):
            neighbors.append(self._random_neighbor_modify_n(c))
        return neighbors

    def random_neighbor(self):
        c = np.random.randint(0, self._c_num)
        return self._random_neighbor_modify_n(c)

    def _random_neighbor_modify_n(self, n: int):
        # Reassign the n-th customer to a new capable facility.
        new_sol_mat = self._mat.copy()
        new_remain = self._f_remain.copy()
        new_target = np.random.randint(0, self._f_num)
        while self._f_remain[new_target] < self._c_demand[n]:
            new_target = np.random.randint(0, self._f_num)
        current_to = new_sol_mat[n].argmax()
        new_remain[current_to] += self._c_demand[n]
        new_remain[new_target] -= self._c_demand[n]
        new_sol_mat[n][current_to] = False
        new_sol_mat[n][new_target] = True  # Select the new target.
        new_sol = Solution(new_sol_mat, self._f_cap, self._c_demand, new_remain)
        return new_sol

    @staticmethod
    def generate_solution(f_num: int, c_num: int, f_cap: np.ndarray, c_demand: np.ndarray):
        # Generate an initial solution.
        sol = np.zeros(shape=(c_num, f_num), dtype=bool)
        # Make a copy of f_cap in order to protect the original data, though it doesn't matter in current code.
        f_remain = f_cap.copy()
        # Randomly assign a customer to a capable facility.
        for c in range(c_num):
            target = np.random.randint(0, f_num)
            while f_remain[target] < c_demand[c]:
                target = np.random.randint(0, f_num)
            sol[c][target] = True
            f_remain[target] -= c_demand[c]
        return Solution(sol, f_cap, c_demand, f_remain)

    def output_result(self, path):
        with open(path, 'w') as f:
            f.write(f'{self._cost}\n')
            f_open = self._mat.sum(axis=0, dtype=bool)
            f.write(' '.join(list(map(lambda x: str(int(x)), f_open))))
            f.write('\n')
            assignment = self._mat.argmax(axis=1)
            f.write(' '.join(list(map(str, assignment))))
