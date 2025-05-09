import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Define three rectangles (R1, R2, R3) for n = 3.
# In our ideal integrality gap instance:
#   • The ILP optimum (maximum disjoint set) is 1 (since every pair of rectangles overlaps enough to forbid selecting two)
#   • The LP relaxation can “fractionally” pick 0.5 from each rectangle, yielding 1.5.
#
# To “break” the common–intersection that would occur with axis–aligned rectangles,
# we leave R1 and R2 axis–aligned and rotate R3 slightly.

fig, ax = plt.subplots(figsize=(7,6))

# R1: axis–aligned rectangle (red)
# bottom-left at (0,0), width=3, height=2
R1 = patches.Rectangle((0, 0), 3, 2, edgecolor='red', facecolor='none', lw=2)
ax.add_patch(R1)
ax.text(1.5, 1, "R1", ha='center', va='center', fontsize=12, color='red')

# R2: axis–aligned rectangle (blue)
# bottom-left at (1,1), width=3, height=2
R2 = patches.Rectangle((1, 1), 3, 2, edgecolor='blue', facecolor='none', lw=2)
ax.add_patch(R2)
ax.text(2.5, 2, "R2", ha='center', va='center', fontsize=12, color='blue')

# R3: a rotated rectangle (green)
# We define it “by hand” as a rectangle with bottom-left at (3, -0.3),
# width=3, height=2, rotated by 15°.
angle = 15  # degrees
R3 = patches.Rectangle((3, -0.3), 3, 2, angle=angle, edgecolor='green', facecolor='none', lw=2)
ax.add_patch(R3)
# To label R3, we can compute its approximate center (this is approximate)
center_R3 = (3 + 3/2, -0.3 + 2/2)
ax.text(center_R3[0], center_R3[1], "R3", ha='center', va='center', fontsize=12, color='green')

# Set plot limits and appearance.
ax.set_xlim(-1, 8)
ax.set_ylim(-2, 6)
ax.set_aspect('equal')
plt.title("Example Instance for n = 3 with Integrality Gap Ratio ≈ 1.5")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.grid(True)
plt.show()
