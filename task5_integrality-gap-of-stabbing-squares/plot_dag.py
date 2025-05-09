###############################################################################
# Plotting Script: Given a bit-sequence, reconstruct and plot squares,
#                  forcing the first bit = 0 in each dimension.
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

def build_rectangles_from_dag_sequence(seq, nRect):
    """
    Convert a bit-sequence (seq) into intervals for one dimension 
    using a left-edge (0) / right-edge (1) picking approach.

    IMPORTANT: We force seq[0] = 0 so that we always pick l1 first.
    """
    if len(seq) > 0:
        seq[0] = 0  # Force the first choice to be the left edge of rect #1

    l_used = [False]*nRect
    r_used = [False]*nRect
    lPos   = [-1]*nRect
    rPos   = [-1]*nRect

    next_l = 0
    index_count = 0

    for bit in seq:
        if bit == 0:
            # pick the next left edge if available
            if next_l < nRect:
                l_used[next_l]  = True
                lPos[next_l]    = index_count
                next_l         += 1
                index_count    += 1
            else:
                # if we've allocated all left edges, do nothing
                pass
        else:
            # pick earliest i s.t. l_i is used but r_i not used
            for i in range(nRect):
                if l_used[i] and not r_used[i]:
                    r_used[i]   = True
                    rPos[i]     = index_count
                    index_count += 1
                    break

    intervals = []
    for i in range(nRect):
        L = lPos[i] if lPos[i] >= 0 else 0
        R = rPos[i] if rPos[i] >= 0 else 0
        if R < L:
            R, L = L, R
        intervals.append( (float(L), float(R)) )
    return intervals


def combine_intervals_as_squares(horizontal_intervals, vertical_intervals):
    """
    Pair up the i-th horizontal interval with the i-th vertical interval => i-th square.
    """
    nRect = len(horizontal_intervals)
    rectangles = []
    for i in range(nRect):
        L = horizontal_intervals[i][0]
        R = horizontal_intervals[i][1]
        T = vertical_intervals[i][0]
        B = vertical_intervals[i][1]
        if T > B: T, B = B, T
        if L > R: L, R = R, L
        rectangles.append( (L, R, T, B) )
    return rectangles


def plot_squares(bit_sequence, nRect, DECISIONS_PER_DIM, show_plot=True, save_path=None):
    """
    Plots squares in different colors, forcing each dimension's first bit = 0.

    bit_sequence    : entire horizontal+vertical bits in a 1D array
    nRect          : number of squares
    DECISIONS_PER_DIM : bits per dimension
    show_plot      : if True, display the figure
    save_path      : if provided, save the figure to this path (e.g., 'myplot.png')
    """
    # Split into horizontal vs. vertical
    seq_horizontal = np.array(bit_sequence[:DECISIONS_PER_DIM], copy=True)
    seq_vertical   = np.array(bit_sequence[DECISIONS_PER_DIM:], copy=True)

    # Build intervals, forcing the first bit to 0 in each dimension
    horiz = build_rectangles_from_dag_sequence(seq_horizontal, nRect)
    vert  = build_rectangles_from_dag_sequence(seq_vertical,   nRect)
    rects = combine_intervals_as_squares(horiz, vert)

    print(horiz, vert)

    if not rects:
        print("No rectangles generated (empty list).")
        return

    # bounding box
    minX = min(r[0] for r in rects)
    maxX = max(r[1] for r in rects)
    minY = min(r[2] for r in rects)
    maxY = max(r[3] for r in rects)

    plt.figure()
    ax = plt.gca()

    # Distinct colors from tab10 (10 unique) or tab20 (20 unique), etc.
    colors = plt.cm.tab10.colors

    for i, (L, R, T, B) in enumerate(rects):
        width  = R - L
        height = B - T
        color  = colors[i % len(colors)]
        # Add rectangle patch, labeled as i+1
        rect_patch = plt.Rectangle(
            (L, T), width, height, fill=False, 
            edgecolor=color, linewidth=2, label=f"Square {i+1}"
        )
        ax.add_patch(rect_patch)

        # Put text label in center
        centerX = (L + R)/2
        centerY = (T + B)/2
        ax.text(centerX, centerY, str(i+1), fontsize=12, 
                ha='center', va='center', color=color)

    # Avoid repeated labels in legend
    handles, labels = ax.get_legend_handles_labels()
    unique_dict = {}
    for h, lab in zip(handles, labels):
        if lab not in unique_dict:
            unique_dict[lab] = h
    ax.legend(unique_dict.values(), unique_dict.keys(), loc='best')

    plt.xlim(minX - 1, maxX + 1)
    plt.ylim(minY - 1, maxY + 1)
    plt.title("Squares from Bit Sequence (forcing first bit=0 in each dimension)")

    if save_path is not None:
        plt.savefig(save_path)
    if show_plot:
        plt.show()


###############################################################################
# Example usage:
###############################################################################
if __name__ == "__main__":
    # Suppose you used nRect=6, DECISIONS_PER_DIM=2*nRect - 3=9 => total bits=18
    # We'll make a random example bit sequence (some might start with 1).
    bits_list = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    # We pass it to plot_squares, which internally sets the first bit to 0 in each dimension.

    bit_array = np.array(bits_list, dtype=int)

    nRect = 6
    DECISIONS_PER_DIM = 9  # for 2*nRect-3
    plot_squares(bit_array, nRect, DECISIONS_PER_DIM, show_plot=True, save_path=None)
