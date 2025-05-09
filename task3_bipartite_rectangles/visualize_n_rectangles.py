import matplotlib.pyplot as plt

def hash_indices(arr):
    index_dict = {}
    for index, value in enumerate(arr):
        if value not in index_dict:
            index_dict[value] = []
        index_dict[value].append(index + 1)
    for key in index_dict:
        index_dict[key] = tuple(index_dict[key])
    return index_dict

H=[4, 1, 3, 1, 2, 2, 4, 3]
V=[1, 4, 2, 4, 3, 1, 3, 2]

result_h = hash_indices(H)
result_v = hash_indices(V)

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

fig, ax = plt.subplots()

for key, coords in rectangles.items():
    bottom_left = coords['bottom_left']
    top_right = coords['top_right']
    width = top_right[0] - bottom_left[0]
    height = top_right[1] - bottom_left[1]
    
    rand_arr = []
    for i in range(1,int((len(H)-1)/2)):
        rand_arr.append(i)
    color = 'red' if key in rand_arr else 'blue'
    
    rect = plt.Rectangle(bottom_left, width, height, fill=None, edgecolor=color)
    ax.add_patch(rect)
    
    ax.text(
        bottom_left[0] + width / 2,
        bottom_left[1] + height / 2,
        str(key),
        fontsize=12,
        color=color,
        ha='center',
        va='center'
    )
ax.set_xlim(0, max(max(result_h.values())[1], max(result_v.values())[1]) + 1)
ax.set_ylim(0, max(max(result_h.values())[1], max(result_v.values())[1]) + 1)
ax.set_xlabel('Horizontal Position')
ax.set_ylabel('Vertical Position')
ax.set_title('Rectangles Plot with Custom Coloring')
plt.grid()
plt.show()