def bubble_multi_transformation(initial_arr, decisions):
    """
    Applies a "multi-action bubble" transformation.
    
    For each adjacent pair, the decision can be:
      0: Do nothing.
      1: Swap the two elements.
      2: Stab: Copy the right element into the left.
    
    Parameters:
      initial_arr: list or numpy array of initial elements.
      decisions: array of integers (0, 1, or 2) with length n*(n-1)//2,
                 following a triangular structure (first pass n-1 decisions,
                 second pass n-2, etc.)
    
    Returns:
      A new list after applying the decisions.
    """
    arr = initial_arr.copy()
    n = len(arr)
    decision_idx = 0
    # For each pass (like in bubble sort, the number of comparisons reduces)
    for pass_num in range(n - 1):
        for i in range(n - 1 - pass_num):
            action = decisions[decision_idx]
            if action == 1:
                # Swap: exchange arr[i] and arr[i+1]
                arr[i], arr[i+1] = arr[i+1], arr[i]
            elif action == 2:
                # Stab: copy the right element into the left, leaving the right unchanged
                arr[i] = arr[i+1]
            # If action==0, do nothing.
            decision_idx += 1
    return arr

# Example usage:
import numpy as np

n = 6
square_size = 2
base = list(range(1, n * square_size, square_size))  # e.g., [1, 3, 5, 7, 9, 11]
# For n=6, we need 6*(6-1)//2 = 15 decisions.
decision_length = n*(n-1)//2
# Here we randomly choose decisions among {0, 1, 2}
decisions = np.random.randint(0, 3, decision_length)
transformed = bubble_multi_transformation(base, decisions)
print("Original:", base)
print("Transformed:", transformed)
