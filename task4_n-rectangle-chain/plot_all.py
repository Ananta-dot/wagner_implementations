import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm

def hash_indices(arr):
    index_dict = {}
    for index, value in enumerate(arr):
        if value not in index_dict:
            index_dict[value] = []
        index_dict[value].append(index + 1)
    for key in index_dict:
        index_dict[key] = tuple(index_dict[key])
    return index_dict

df = pd.read_csv("results.csv")

for idx, row in df.iterrows():
    H_str = row['H']
    V_str = row['V']
    H_list = [int(x.strip()) for x in H_str.split(',')]
    V_list = [int(x.strip()) for x in V_str.split(',')]
    
    result_h = hash_indices(H_list)
    result_v = hash_indices(V_list)
    
    rectangles = {}
    for key in result_h.keys():
        h_start, h_end = result_h[key]
        v_start, v_end = result_v[key]
        rectangles[key] = {
            'bottom_left': (h_start, v_start),
            'top_right': (h_end, v_end)
        }
    
    fig, ax = plt.subplots()
    
    colors = cm.rainbow(np.linspace(0, 1, len(rectangles)))
    
    for i, (key, coords) in enumerate(rectangles.items()):
        bottom_left = coords['bottom_left']
        top_right = coords['top_right']
        width = top_right[0] - bottom_left[0]
        height = top_right[1] - bottom_left[1]
        color = colors[i]
        
        rect = plt.Rectangle(bottom_left, width, height, fill=True,
                             facecolor=color, edgecolor='black', alpha=0.6)
        ax.add_patch(rect)
        
        ax.text(bottom_left[0] + width / 2, bottom_left[1] + height / 2,
                str(key), ha='center', va='center', color='black', fontsize=12)
    
    max_h = max(val[1] for val in result_h.values())
    max_v = max(val[1] for val in result_v.values())
    max_val = max(max_h, max_v) + 1
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel('Horizontal Position')
    ax.set_ylabel('Vertical Position')
    ax.set_title(f"Rectangles (max(H) = {max(H_list)})")
    plt.grid(True)
    
    filename = f"n_chain_{max(H_list)}.png"
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot for row {idx} as {filename}")
