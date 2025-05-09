import matplotlib.pyplot as plt

from networkx import hits
import numpy as np
import pandas as pd


bits= [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0]

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

def build_base_array(n_):
    arr = []
    for val in range(1, n_ + 1):
        arr.extend([val, val])
    return np.array(arr, dtype=int)

base_arr = build_base_array(10)  

def apply_comps(baseA, bits, comps):
    arrc = baseA.copy()
    for i, bit in enumerate(bits):
        if bit == 1:
            a, b = comps[i]
            arrc[a], arrc[b] = arrc[b], arrc[a]
    return arrc

comps = get_batcher_oe_comparators_py(len(base_arr)) 
m = len(comps)

s1 = bits[:m]    
s2 = bits[m:2*m] 

# arrH = [3, 8, 6, 8, 4, 5, 6, 1, 1, 2, 7, 3, 2, 4, 5, 7]
# arrV = [4, 8, 7, 1, 6, 4, 6, 3, 3, 8, 2, 7, 5, 5, 1, 2]

arrH = build_base_array(10)
arrV = build_base_array(10)
# for i in arrH_old:
#     if i < 3:
#         arrH.append(i)
#     else:
#         arrH.append(i-1)
# for i in arrV_old:
#     if i < 3:
#         arrV.append(i)
#     else:
#         arrV.append(i-1)

arrH = np.array(arrH)
arrV = np.array(arrV)

print("Horizontal ordering:", arrH)
print("Vertical ordering:  ", arrV)

def hash_indices(arr):
    index_dict = {}
    for index, value in enumerate(arr):
        if value not in index_dict:
            index_dict[value] = []
        index_dict[value].append(index + 1)
    for key in index_dict:
        index_dict[key] = tuple(index_dict[key])
    return index_dict

result_h = hash_indices(arrH)
result_v = hash_indices(arrV)

# result_h = arrH
# result_v = arrV

print(result_h)
print(result_v)

rectangles = {}
for key in result_h.keys():
    h_start, h_end = result_h[key]
    v_start, v_end = result_v[key]
    rectangles[key] = {
        'bottom_left': (h_start, v_start),
        'top_right': (h_end, v_end),
    }
color_list = ['red', 'orange', 'blue', 'cyan', 'yellow', 'green', 'magenta', 'purple']
fig, ax = plt.subplots()

for i, (key, coords) in enumerate(rectangles.items()):
    bottom_left = coords['bottom_left']
    top_right = coords['top_right']
    width = top_right[0] - bottom_left[0]
    height = top_right[1] - bottom_left[1]
    
    color = color_list[i % len(color_list)]
    
    rect = plt.Rectangle(bottom_left, width, height, fill=True, facecolor=color, edgecolor=color)
    ax.add_patch(rect)
    
    ax.text(
        bottom_left[0] + width / 2,
        bottom_left[1] + height / 2,
        str(key),
        fontsize=12,
        color='white',
        ha='center',
        va='center'
    )
max_val = max(
    max(val[1] for val in result_h.values()),
    max(val[1] for val in result_v.values())
) + 1

ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.set_xlabel('Horizontal Position')
ax.set_ylabel('Vertical Position')
ax.set_title('Rectangles Plot with Different Colors')
plt.grid()
plt.show()
