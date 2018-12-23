from itertools import islice
from time import time

import numpy as np
import math

from solution import Solution


def simulated_annealing(solution: Solution, f_cost: np.ndarray, c_cost: np.ndarray, output_path: str):
    initial_acceptance = .95
    k = 0
    max_k = 1000
    best_cost = solution.cost(f_cost, c_cost)

    # Set an initial temperature by making 95% non-improving accepted at first.
    init_cost_sum = 0
    i = 0
    while i < 100:
        neighbor = solution.random_neighbor()
        neighbor_cost = neighbor.cost(f_cost, c_cost)
        if neighbor_cost <= best_cost:
            continue
        init_cost_sum += neighbor_cost
        i += 1

    init_cost_ave = init_cost_sum / 100
    init_delta_e = init_cost_ave - best_cost
    t0 = -init_delta_e / math.log(1 / initial_acceptance - 1)
    while k <= max_k:
        temperature = t0 / (1 + math.log(1 + k))
        for inner in range(500):
            neighbor = solution.random_neighbor()
            delta_e = neighbor.cost(f_cost, c_cost) - best_cost
            pk = 1 / (1 + math.e ** (delta_e / temperature))
            if delta_e <= 0 or np.random.random_sample() < pk:
                solution = neighbor
                best_cost += delta_e
        # print(f'{k:4}: {best_cost} T0: {t0:.2f} T: {temperature:.2f}')
        k += 1
    solution.output_result(output_path)
    return best_cost


def hill_climbing(solution: Solution, f_cost: np.ndarray, c_cost: np.ndarray, output_path: str):
    tries = 0
    max_tries = 50000
    best_cost = solution.cost(f_cost, c_cost)

    while tries < max_tries:
        moved = False
        for neighbor in solution.all_neighbors():
            new_cost = neighbor.cost(f_cost, c_cost)
            if new_cost < best_cost:
                solution = neighbor
                best_cost = new_cost
                # print(f'{tries}: {best_cost}')
                moved = True
        tries += 1
        if not moved:
            # print('Not moved.')
            break
    solution.output_result(output_path)
    return best_cost


def strip_dot_then_to_int(x: str):
    return int(x.strip('. '))


if __name__ == '__main__':
    print('||hill-climbing result|hill-climbing time (s)|SA result|SA time (s)|')
    print('|-|-|-|-|-|')
    for i in range(67, 72):
        # Input data
        with open(f'Instances/p{i}') as f:
            facility_num, customer_num = map(int, f.readline().split())
            facility_capacity, facility_cost = np.empty(facility_num, dtype=int), np.empty(facility_num, dtype=int)
            customer_demand = np.empty(customer_num, dtype=int)
            customer_cost = np.empty((customer_num, facility_num), dtype=int)

            for j in range(facility_num):
                facility_capacity[j], facility_cost[j] = map(int, f.readline().split())


            # The given input file is formatted to 'n * 10' shape rather than 'm * customer_num' shape.
            # Do this to parse all desired numbers at once.
            # The strange if_else statement is used to deal with inconsistently formatted file p67.
            line = ''.join(islice(f, customer_num // 10 if i != 67 else customer_num // 10 * 2))
            customer_demand[...] = list(map(strip_dot_then_to_int, line.split()))

            for j in range(customer_num):
                # The strange if_else statement is used to deal with inconsistently formatted file p67.
                line = ''.join(islice(f, facility_num // 10 if i != 67 else facility_num // 10 * 2))
                customer_cost[j] = list(map(strip_dot_then_to_int, line.split()))

            init_sol = Solution.generate_solution(facility_num, customer_num, facility_capacity, customer_demand)

            start = time()
            hc = hill_climbing(init_sol, facility_cost, customer_cost, f'hc/p{i}')
            hc_end = time()
            sa = simulated_annealing(init_sol, facility_cost, customer_cost, f'sa/p{i}')
            sa_end = time()

            print(f'|p{i}|{hc}|{hc_end - start:.2f}|{sa}|{sa_end - hc_end:.2f}|')
