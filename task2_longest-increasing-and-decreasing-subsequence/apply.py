import numpy as np
import random

def get_batcher_oe_comparators(n):
    comparators = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            j_start = k % p
            j_end = n - 1 - k
            step_j = 2 * k
            j = j_start
            while j <= j_end:
                i_max = min(k - 1, n - j - k - 1)
                for i in range(i_max + 1):
                    left = (i + j) // (2 * p)
                    right = (i + j + k) // (2 * p)
                    if left == right:
                        comparators.append((i + j, i + j + k))
                j += step_j
            k //= 2
        p *= 2
    return comparators

def lengthOfLIS(nums):
    n = len(nums)
    dp = [1]*n
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp)

def lengthOfLDS(nums):
    nums.reverse()
    return lengthOfLIS(nums)

n = 8
comparators_arr = get_batcher_oe_comparators(n)

def apply_bits_and_get_array(bits):
    """
    Replicates the logic of 'calc_score':
      - create a random permutation,
      - apply compare-and-swap for each comparator bit=1,
      - return the final array.
    """
    arr = list(range(n))  # arr = [0,1,2,3,4,5,6,7]
    # shuffle in place (Fisher-Yates)
    for i in range(n - 1):
        j = i + int(random.random() * (n - i))
        arr[i], arr[j] = arr[j], arr[i]
    
    print(arr)
    # apply chosen comparators
    for idx, b in enumerate(bits):
        if b == 1:
            a, c = comparators_arr[idx]
            arr[a], arr[c] = arr[c], arr[a]
    return arr

some_bits = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]

final_arr = apply_bits_and_get_array(some_bits)
print("Random permutation after applying bits =>", final_arr)
print("Length of LIS =>", lengthOfLIS(final_arr))
print("Length of LDS =>", lengthOfLDS(final_arr))
print("N - max(LIS, LDS)=", n-max(lengthOfLIS(final_arr),lengthOfLDS(final_arr)))


