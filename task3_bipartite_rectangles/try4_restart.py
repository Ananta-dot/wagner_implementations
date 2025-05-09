import numpy as np
import random as rn

def get_batcher_oe_comparators_py(n):
    comparators = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            j_start = k % p
            j_end = n - 1 - k
            step_j = 2 * k
            j = j_start
            while j <= j_end:
                i_max = min(k - 1, n - j - k - 1)
                for i in range(i_max + 1):
                    left = (i + j) // (2 * p)
                    right = (i + j + k) // (2 * p)
                    if left == right:
                        comparators.append((i + j, i + j + k))
                j += step_j
            k //= 2
        p *= 2
    return comparators

def apply_comparators(arr, state, comparators_arr):
    arr_copy = arr.copy()
    DECISIONS = len(state)
    for i in range(DECISIONS):
        if state[i] == 1:
            a = comparators_arr[i][0]  
            b = comparators_arr[i][1]
            arr_copy[a], arr_copy[b] = arr_copy[b], arr_copy[a]
    return arr_copy

n = 4

def build_arr(): 
    chunk = []
    for val in range(1, n + 1):
        chunk.extend([val, val]) 
    return np.array(chunk, dtype=np.int64), np.array(chunk, dtype=np.int64) 

base_h_arr, base_v_arr = build_arr()

# print(base_h_arr)
# print(base_v_arr)

comparators = get_batcher_oe_comparators_py(n*2)
# print(comparators)

states = np.random.randint(0, 2, size=len(comparators)).tolist()
# print(states)

base_h_arr = apply_comparators(base_h_arr, states, comparators)
base_v_arr = apply_comparators(base_v_arr, states, comparators)

print(base_h_arr, base_v_arr)

