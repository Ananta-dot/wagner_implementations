import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def hash_indices(arr):
    index_dict = {}
    for index, value in enumerate(arr):
        if value not in index_dict:
            index_dict[value] = []
        index_dict[value].append(index + 1)
    for key in index_dict:
        index_dict[key] = tuple(index_dict[key])
    return index_dict

def intersect_interval_strict(ab, cd):
    a, b = sorted(ab)
    c, d = sorted(cd)
    if b <= c or d <= a:
        return False
    return True

def intersect_strict(rectA, rectB):
    return (intersect_interval_strict(rectA[0], rectB[0]) and
            intersect_interval_strict(rectA[1], rectB[1]))



H = [2, 4, 1, 3, 5, 5, 6, 6, 8, 8, 7, 7, 2, 4, 1, 3]
V = [8, 5, 6, 7, 2, 2, 4, 4, 1, 1, 3, 8, 3, 6, 7, 5]
n = len(H)/4 # 4  # given

result_h = hash_indices(H)
result_v = hash_indices(V)
print("Horizontal ranges:", result_h)
print("Vertical ranges:", result_v)

keys = sorted(result_h.keys())

rects = []
for key in keys:
    rects.append((result_h[key], result_v[key]))
colors = []
for i, key in enumerate(keys):
    if i < n:
        colors.append(0)
    else:
        colors.append(1)

print("\nConstructed rectangles:")
for i, r in enumerate(rects):
    print(f"  Rect{i+1}: {r}, color: {'red' if colors[i]==0 else 'blue'}")

score = 0
num_rects = len(rects)
for i in range(num_rects):
    for j in range(i+1, num_rects):
        same_color = (colors[i] == colors[j])
        does_intersect = intersect_strict(rects[i], rects[j])
        if same_color:
            if not does_intersect:
                score += 1
        else:
            if does_intersect:
                score += 1

MAX_SCORE = n**2 + n*(n-1)

print(f"\nFinal Score = {score} (max theoretical score = {MAX_SCORE})")

fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect('equal')
ax.set_title(f"rectangles ([1...{n}] = red), ([{n+1}...{2*n}] = blue)")

min_x = min(r[0][0] for r in rects)
max_x = max(r[0][1] for r in rects)
min_y = min(r[1][0] for r in rects)
max_y = max(r[1][1] for r in rects)

for i, r in enumerate(rects):
    x1, x2 = sorted(r[0])
    y1, y2 = sorted(r[1])
    width = x2 - x1
    height = y2 - y1
    col = 'red' if colors[i] == 0 else 'blue'
    rect_patch = Rectangle((x1, y1), width, height, edgecolor='black', facecolor=col, alpha=0.4)
    ax.add_patch(rect_patch)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    ax.text(cx, cy, f"#{i+1}", fontsize=12, ha='center', va='center', color='k', fontweight='bold')

ax.set_xlim(min_x-1, max_x+1)
ax.set_ylim(min_y-1, max_y+1)
ax.set_xlabel("relative pos (horizontal)")
ax.set_ylabel("relative pos (vertical)")
ax.grid(True)
plt.show()