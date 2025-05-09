def get_batcher_oe_comparators(n):
    """
    pseudocode from wikiperdia (modified to just add comparators instead of swapping)
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
            j_start = k % p  # why important? because k = 0 when k == p instead so we start over!
            j_end = n - 1 - k 
            step_j = 2 * k
            j = j_start
            while j <= j_end:
                i_max = min(k - 1, n - j - k - 1)
                for i in range(i_max + 1):  # range is exclusive at the end.
                    left = (i + j) // (2 * p)
                    right = (i + j + k) // (2 * p)
                    if left == right:
                        comparators.append((i + j, i + j + k))
                j += step_j
            k //= 2  # integer division
        p *= 2
    return comparators

n = 8
comps = get_batcher_oe_comparators(n)
print(f"Batcher Odd-Even Mergesort comparators for n = {n}:")
for comp in comps:
    print(comp)
print(len(comps))
