import numpy as np
from pulp import *

def calc_score(bits, n):
    m = len(bits) // 2
    s1 = bits[:m]
    s2 = bits[m:]
    
    def build_base_array(n_):
        return np.repeat(np.arange(1, n_+1), 2)
    
    def apply_comps(baseA, bits, comps):
        arrc = baseA.copy()
        for i, bit in enumerate(bits):
            if bit == 1:
                a, b = comps[i]
                arrc[a], arrc[b] = arrc[b], arrc[a]
        return arrc
    
    def get_batcher_oe_comparators_py(n_):
        comps = []
        p = 1
        while p < n_:
            k = p
            while k >= 1:
                j_start = k % p
                j_end = n_ - 1 - k
                step_j = 2 * k
                j = j_start
                while j <= j_end:
                    i_max = min(k - 1, n_ - j - k - 1)
                    for i in range(i_max + 1):
                        left = (i + j) // (2 * p)
                        right = (i + j + k) // (2 * p)
                        if left == right:
                            comps.append((i + j, i + j + k))
                    j += step_j
                k //= 2
            p *= 2
        return comps
    
    base_len = 2 * n
    comps = get_batcher_oe_comparators_py(base_len)
    base_arr = build_base_array(n)
    arrH = apply_comps(base_arr, s1, comps)
    arrV = apply_comps(base_arr, s2, comps)
    
    rectangles = []
    for i in range(1, n+1):
        x_indices = np.where(arrH == i)[0]
        y_indices = np.where(arrV == i)[0]
        if len(x_indices) < 2 or len(y_indices) < 2:
            rectangles.append(((0,0), (0,0)))
        else:
            x1, x2 = min(x_indices), max(x_indices)
            y1, y2 = min(y_indices), max(y_indices)
            rectangles.append(((x1, x2), (y1, y2)))
    
    def solve_disjoint_rectangles(rectangles):
        n_rect = len(rectangles)
        constraints = []
        
        for i in range(n_rect):
            (x1i, x2i), (y1i, y2i) = rectangles[i]
            for j in range(i+1, n_rect):
                (x1j, x2j), (y1j, y2j) = rectangles[j]
                if (x1i < x2j and x1j < x2i and y1i < y2j and y1j < y2i):
                    constraints.append([i, j])
        
        
        prob = LpProblem("MaxDisjointRectangles", LpMaximize)
        x = [LpVariable(f"x{i}", cat="Binary") for i in range(n_rect)]
        
        prob += lpSum(x)
        
        for i, j in constraints:
            prob += x[i] + x[j] <= 1
        
        prob.solve(PULP_CBC_CMD(msg=0))
        
        return value(prob.objective), len(x) - len(constraints)
    
    lp_val, ilp_val = solve_disjoint_rectangles(rectangles)
    
    if ilp_val < 1e-9:
        ratio = 0.0
    else:
        ratio = lp_val / ilp_val
        
    return ratio

# Example usage
bits = [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
n = 14  # Assuming n = 14 as in the original script

score = calc_score(bits, n)
print(f"Score: {score}")
