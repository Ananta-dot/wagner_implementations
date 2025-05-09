import numpy as np

n = 3
lst_r = [x for x in range(1, n+1) for _ in range(2)] * 2
print(lst_r)

lst_b = [x for x in range(1, n+1) for _ in range(2)] * 2
print(lst_b)

# def get_batcher_oe_comparators_py(n):
#     comparators = []
#     p = 1
#     while p < n:
#         k = p
#         while k >= 1:
#             j_start = k % p
#             j_end = n - 1 - k
#             step_j = 2 * k
#             j = j_start
#             while j <= j_end:
#                 i_max = min(k - 1, n - j - k - 1)
#                 for i in range(i_max + 1):
#                     left = (i + j) // (2 * p)
#                     right = (i + j + k) // (2 * p)
#                     if left == right:
#                         comparators.append((i + j, i + j + k))
#                 j += step_j
#             k //= 2
#         p *= 2
#     return comparators

# comparators_arr = get_batcher_oe_comparators_py(n * 8)

# def apply_comparators(arr, state, comparators_arr):
#     arr_copy = arr.copy()
#     DECISIONS = len(state)
#     for i in range(DECISIONS):
#         if state[i] == 1:
#             a = comparators_arr[i][0]  
#             b = comparators_arr[i][1]
#             arr_copy[a], arr_copy[b] = arr_copy[b], arr_copy[a]
#     return arr_copy

# states = np.random.randint(0, 2, size=len(lst)).tolist()

# arr = apply_comparators(lst, states, comparators_arr)

# print(states)
# print(arr)