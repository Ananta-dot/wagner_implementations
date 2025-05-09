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

H = [3, 2, 4, 3, 5, 4, 6, 1, 5, 1, 6, 2]
V = [5, 3, 4, 6, 5, 1, 4, 6, 2, 2, 1, 3]

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
