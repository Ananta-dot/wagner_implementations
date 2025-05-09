def get_batcher_oe_comparators(n):
    """
    Generate a list of comparator pairs for Batcher's Odd-Even Mergesort 
    for an input of n elements.
    
    The pseudocode implemented:
      for p = 1, 2, 4, 8, ...  (while p < n)
        for k = p, p/2, p/4, ...  (while k >= 1)
          for j = (k mod p) to (n - 1 - k) with a step size of 2*k
            for i = 0 to min(k-1, n - j - k - 1) with a step of 1
              if floor((i+j)/(2*p)) == floor((i+j+k)/(2*p))
                add comparator (i+j, i+j+k)
    """
    comparators = []

    p = 1
    while p < n:
        k = p
        while k >= 1:
            j_start = k % p  # For k < p, this equals k.
            j_end = n - 1 - k  # j goes up to n-1-k (inclusive).
            step_j = 2 * k
            j = j_start
            while j <= j_end:
                i_max = min(k - 1, n - j - k - 1)
                for i in range(i_max + 1):  # +1 since range end is exclusive.
                    left = (i + j) // (2 * p)
                    right = (i + j + k) // (2 * p)
                    if left == right:
                        comparators.append((i + j, i + j + k))
                j += step_j
            k //= 2  # Halve k (using integer division)
        p *= 2  # Double p
    return comparators


def batcher_sort(arr):
    """
    Sort a list 'arr' using the comparators from the Batcher odd-even mergesort network.
    
    This function assumes that len(arr) = n and uses get_batcher_oe_comparators(n)
    to generate the comparator network. Then, it applies each comparator in order,
    performing a compare-and-swap operation.
    """
    n = len(arr)
    comparators = get_batcher_oe_comparators(n)
    # Print the comparator pairs for reference (optional)
    print("Comparator pairs:")
    for c in comparators:
        print(c)
    
    # Apply each comparator sequentially.
    for (i, j) in comparators:
        if arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]
    return arr


# -----------------------
# Example usage:
if __name__ == "__main__":
    # Let's define an unsorted list of 8 numbers.
    unsorted_list = [7, 3, 6, 4, 1]#, 2, 8, 5]
    print("Unsorted list:", unsorted_list)
    
    # Use the batcher_sort function to sort the list.
    sorted_list = batcher_sort(unsorted_list.copy())
    print("Sorted list:  ", sorted_list)
