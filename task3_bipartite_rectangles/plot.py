import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_rectangles():
    """
    Plot the four rectangles from the final array:
      - Rectangle 0 -> (2,4,2,4)
      - Rectangle 1 -> (3,1,3,1)
      - Rectangle 2 -> (3,1,4,2)
      - Rectangle 3 -> (1,4,3,2)
    """
    # Interpret each tuple as (x_left, x_right, y_bottom, y_top).
    rects = [
        ((2,4,2,4), "Rect0 (blue)"),  # We'll assume rect0 is blue
        ((3,1,3,1), "Rect1 (blue)"),
        ((3,1,4,2), "Rect2 (red)"),
        ((1,4,3,2), "Rect3 (red)"),
    ]
    
    # Colors to use: first two = blue, last two = red
    colors = ["blue", "blue", "red", "red"]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    for idx, (coords, label) in enumerate(rects):
        x_left, x_right, y_bottom, y_top = coords
        left   = min(x_left, x_right)
        right  = max(x_left, x_right)
        bottom = min(y_bottom, y_top)
        top    = max(y_bottom, y_top)
        
        # Update plot bounds
        min_x = min(min_x, left)
        max_x = max(max_x, right)
        min_y = min(min_y, bottom)
        max_y = max(max_y, top)
        
        width  = right - left
        height = top - bottom
        
        rect_patch = Rectangle(
            (left, bottom),
            width, height,
            facecolor=colors[idx],
            alpha=0.3,
            edgecolor="black",
            linewidth=2
        )
        ax.add_patch(rect_patch)
        
        # Label near the center
        cx = (left + right) / 2
        cy = (bottom + top) / 2
        ax.text(cx, cy, label,
                ha='center', va='center',
                fontsize=10, fontweight='bold')

    # Pad the axes a bit
    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Visualization of 4 Rectangles (2 Blue, 2 Red)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_rectangles()
