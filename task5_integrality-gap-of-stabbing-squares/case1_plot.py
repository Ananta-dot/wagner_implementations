import matplotlib.pyplot as plt

def decode_ordering(bits, n):
    """
    Decode a bitstring into a valid ordering of rectangle events.
    Each rectangle has two events: left (l) and right (r).
    The bitstring determines the order of these events.

    Args:
        bits (list): A binary list representing the bitstring.
        n (int): Number of rectangles.

    Returns:
        list: Decoded ordering of events.
    """
    base = [i // 2 + 1 for i in range(2 * n)]  # Initial order: [1, 1, 2, 2, ..., n, n]
    comps = [(i, i + 1) for i in range(len(base) - 1)]  # Comparators for adjacent swaps

    for i, bit in enumerate(bits):
        if bit == 1:
            a, b = comps[i]
            base[a], base[b] = base[b], base[a]

    return base

def create_rectangles(h_order, v_order, n):
    """
    Create rectangles based on horizontal and vertical orderings.

    Args:
        h_order (list): Horizontal ordering of events.
        v_order (list): Vertical ordering of events.
        n (int): Number of rectangles.

    Returns:
        list: List of rectangles with their boundaries.
    """
    rectangles = []
    for i in range(1, n + 1):
        h_pos = [idx for idx, val in enumerate(h_order) if val == i]
        v_pos = [idx for idx, val in enumerate(v_order) if val == i]

        if len(h_pos) < 2 or len(v_pos) < 2:
            return None  # Invalid configuration

        x1, x2 = sorted(h_pos)[0], sorted(h_pos)[1]
        y1, y2 = sorted(v_pos)[0], sorted(v_pos)[1]
        rectangles.append(((x1, x2 + 1), (y1, y2 + 1)))

    return rectangles

def plot_rectangles_with_lines(rectangles):
    """
    Plot rectangles and their corresponding top and right lines.

    Args:
        rectangles (list): List of rectangles with their boundaries.
    """
    plt.figure(figsize=(8, 6))

    for i, ((x1, x2), (y1, y2)) in enumerate(rectangles):
        # Plot the rectangle
        plt.fill_between([x1, x2], y1, y2, alpha=0.3, label=f'Rect {i+1}')

        # Plot the top line
        plt.plot([x1, x2], [y2, y2], 'b--', alpha=0.7, label=f'Top Line {i+1}' if i == 0 else "")

        # Plot the right line
        plt.plot([x2, x2], [y1, y2], 'r--', alpha=0.7, label=f'Right Line {i+1}' if i == 0 else "")

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Rectangles with Top and Right Lines')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
n =   # Number of rectangles
bits = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0] 

# Decode orderings
horizontal_order = decode_ordering(bits[:len(bits)//2], n)
vertical_order = decode_ordering(bits[len(bits)//2:], n)

# Create rectangles
rectangles = create_rectangles(horizontal_order[:8], vertical_order[:8], n)

# Plot rectangles with top and right lines
if rectangles:
    plot_rectangles_with_lines(rectangles)
else:
    print("Invalid rectangle configuration!")
