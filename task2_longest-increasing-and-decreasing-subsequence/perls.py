import math

def batcher_oe_comparators(n):
    comparators = []
    t = math.ceil(math.log2(n))
    q = 2 ** (t - 1)
    r = 0
    d = 1

    while d > 0:
        for i in range(n - d):
            if (i & 1) == r:
                comparators.append((i, i + d))
        
        d = q - 1
        q = q // 2
        r = 1
    
    # Remove duplicate comparisons and sort pairs
    comparators = sorted(set(tuple(sorted(pair)) for pair in comparators))
    return comparators

print(batcher_oe_comparators(4))