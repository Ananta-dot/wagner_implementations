from typing import List
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ────────────────────────────────────────────────────────────────────────────────
# Rectangle geometry utilities
# ────────────────────────────────────────────────────────────────────────────────
class Rectangle:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, special: int = None):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.special = special          # used while building the recursive gadget

    def intersects(self, other: "Rectangle") -> bool:
        """Open-interval (strict) intersection test."""
        return not (
            self.x2 <= other.x1 or self.x1 >= other.x2 or
            self.y2 <= other.y1 or self.y1 >= other.y2
        )

    # Nice printable form
    def __repr__(self):
        return f"Rect({self.x1:.2f},{self.y1:.2f},{self.x2:.2f},{self.y2:.2f}, special={self.special})"


# ────────────────────────────────────────────────────────────────────────────────
# Low–level helpers
# ────────────────────────────────────────────────────────────────────────────────
def rotate_rectangles(rectangles: List[Rectangle]) -> List[Rectangle]:
    """
    90° clockwise rotation around origin (x, y) ↦ (y, −x) *with bounding-box re-ordering*.
    The returned list is a fresh copy; the originals are untouched.
    """
    rotated = []
    for rect in rectangles:
        new_x1, new_y1 = rect.y1, -rect.x2
        new_x2, new_y2 = rect.y2, -rect.x1
        x1, x2 = min(new_x1, new_x2), max(new_x1, new_x2)
        y1, y2 = min(new_y1, new_y2), max(new_y1, new_y2)
        rotated.append(Rectangle(x1, y1, x2, y2, special=rect.special))
    return rotated


def translate_rectangles(rectangles: List[Rectangle], dx: float, dy: float) -> None:
    """In-place translation by (dx, dy)."""
    for rect in rectangles:
        rect.x1 += dx;  rect.x2 += dx
        rect.y1 += dy;  rect.y2 += dy


def bounding_box(rectangles: List[Rectangle]):
    min_x = min(r.x1 for r in rectangles)
    max_x = max(r.x2 for r in rectangles)
    min_y = min(r.y1 for r in rectangles)
    max_y = max(r.y2 for r in rectangles)
    return min_x, min_y, max_x, max_y


# ────────────────────────────────────────────────────────────────────────────────
# Recursive construction of Iₙ instance
# ────────────────────────────────────────────────────────────────────────────────
def build_rectangles(n: int) -> List[Rectangle]:
    """
    Build the gadget Iₙ from the paper (recursive L-shape of rectangles).
    The base case (n = 1) is hard-coded; each level adds 3 outer rectangles
    around a rotated copy of the previous instance.
    """
    rects = [
        Rectangle(-10, -2,  -6,  8, special=1),   # R₁₁ (left)
        Rectangle(-8,   3,   6,  8, special=3),   # R₁₃ (top)
        Rectangle( 2,  -8,   6,  4, special=2),   # R₁₂ (right)
        Rectangle(-8,  -8,   6, -4),              # interior bottom
        Rectangle(-10, -8,  -7, -1)               # interior bottom-left
    ]

    # add levels 2 … n
    for level in range(2, n + 1):
        # rotate old instance
        rotated = rotate_rectangles(rects)
        min_x, min_y, max_x, max_y = bounding_box(rotated)
        translate_rectangles(rotated, dx=-min_x, dy=-min_y)  # move to positive quadrant

        # identify special rectangles in the rotated copy
        R1_rot = next(r for r in rotated if r.special == 1)
        R2_rot = next(r for r in rotated if r.special == 2)
        R3_rot = next(r for r in rotated if r.special == 3)

        # after copying in, forget “special” flags to avoid clashing in the next round
        for r in rotated:
            r.special = None

        min_x, min_y, max_x, max_y = bounding_box(rotated)
        span_x = max_x - min_x
        span_y = max_y - min_y
        overlap = 1.0   # how much the outer frame overlaps the rotated core

        # new left, right, top frame rectangles (mark them special for next iteration)
        left_rect = Rectangle(
            min_x - 0.25 * span_x,           # x1
            min(R2_rot.y1, R3_rot.y1),       # y1
            min_x + overlap,                 # x2
            max(R2_rot.y2, R3_rot.y2),       # y2
            special=1
        )
        # clip so it doesn't intrude into rotated R1
        if left_rect.intersects(R1_rot):
            if R1_rot.y1 < left_rect.y2 < R1_rot.y2:
                left_rect.y2 = R1_rot.y1
            if R1_rot.y1 < left_rect.y1 < R1_rot.y2:
                left_rect.y1 = R1_rot.y2

        right_rect = Rectangle(
            max_x - overlap,                         # x1
            min(R1_rot.y1, R2_rot.y1),               # y1
            max_x + 0.25 * span_x,                   # x2
            max(R1_rot.y2, R2_rot.y2),               # y2
            special=2
        )
        if right_rect.intersects(R3_rot):
            if R3_rot.y1 < right_rect.y2 < R3_rot.y2:
                right_rect.y2 = R3_rot.y1
            if R3_rot.y1 < right_rect.y1 < R3_rot.y2:
                right_rect.y1 = R3_rot.y2

        top_rect = Rectangle(
            left_rect.x1,                    # x1
            max_y - overlap,                 # y1
            right_rect.x2,                   # x2
            max_y + 0.25 * span_y,           # y2
            special=3
        )

        # new instance
        rects = rotated + [left_rect, right_rect, top_rect]

    return rects


# ────────────────────────────────────────────────────────────────────────────────
# Computing the maximum independent set (exact brute force)
# ────────────────────────────────────────────────────────────────────────────────
def max_independent_set_size(rectangles: List[Rectangle]) -> int:
    """Exact MIS via brute-force enumeration (2^m subsets).  OK for small m."""
    n = len(rectangles)
    best = 0
    for mask in range(1 << n):
        chosen = []
        valid = True
        for i in range(n):
            if mask & (1 << i):
                # check intersection with those already chosen
                if any(rectangles[i].intersects(r) for r in chosen):
                    valid = False
                    break
                chosen.append(rectangles[i])
        if valid:
            best = max(best, len(chosen))
    return best


# ────────────────────────────────────────────────────────────────────────────────
# Pretty plotting
# ────────────────────────────────────────────────────────────────────────────────
def plot_rectangles(rectangles: List[Rectangle],
                    title: str = "Rectangles",
                    shift_to_positive: bool = True):
    """
    Plot rectangles with unique colours (tab20).  If any coordinate is negative,
    temporarily translate so the displayed axes start at (0, 0).
    """
    # 1) bounding box & shift
    min_x, min_y, max_x, max_y = bounding_box(rectangles)
    dx = -min_x if shift_to_positive and min_x < 0 else 0
    dy = -min_y if shift_to_positive and min_y < 0 else 0

    cmap = cm.get_cmap("tab20")
    fig, ax = plt.subplots()

    for idx, rect in enumerate(rectangles):
        x1_plot = rect.x1 + dx
        y1_plot = rect.y1 + dy
        width  = rect.x2 - rect.x1
        height = rect.y2 - rect.y1
        colour = cmap(idx % 20)

        patch = plt.Rectangle(
            (x1_plot, y1_plot), width, height,
            facecolor=colour, edgecolor="black",
            alpha=0.35, linewidth=1.5
        )
        ax.add_patch(patch)
        ax.text(x1_plot + width / 2,
                y1_plot + height / 2,
                f"R{idx}",
                ha="center", va="center", fontsize=9)

    margin = 1.0
    ax.set_xlim(0, (max_x - min_x) + margin)
    ax.set_ylim(0, (max_y - min_y) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("Horizontal axis")
    ax.set_ylabel("Vertical axis")
    plt.show()


# ────────────────────────────────────────────────────────────────────────────────
# Demo loop — build Iₙ, compute integrality-gap ratio, plot, *print coordinates*
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for n in range(1, 7):
        rects = build_rectangles(n)
        total = len(rects)
        mis   = max_independent_set_size(rects)
        fractional_opt = 0.5 * total
        gap_ratio = fractional_opt / mis

        print(f"\n=== Instance n = {n} ===")
        print(f"Rectangles: {total}   MIS: {mis}   fractional_opt: {fractional_opt}   gap: {gap_ratio:.3f}")

        # Plot
        plot_rectangles(rects, title=f"n = {n}, gap = {gap_ratio:.3f}")

        # Print all coordinates after the plot
        print("Rectangle coordinates:")
        for idx, r in enumerate(rects):
            print(f"  R{idx:02d}: ({r.x1:.2f}, {r.y1:.2f}) → ({r.x2:.2f}, {r.y2:.2f})")
