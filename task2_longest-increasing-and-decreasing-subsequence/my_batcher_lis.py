#######################################
# my_batcher_lis.py
#######################################

import numpy as np
import random

def get_batcher_oe_comparators(n):
    """
    Pseudocode from Wikipedia (modified to just add comparators instead of swapping).
    Returns a list of (i, j) pairs representing comparators.
    """
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

def lengthOfLIS(nums) -> int:
    """
    Standard O(n^2) dynamic programming approach to computing
    the length of the longest increasing subsequence.
    """
    if not nums:
        return 0
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# Pick an N (number of elements). Adjust as needed.
N = 8

# Generate the comparators for Batcher's odd-even mergesort network
comparators = get_batcher_oe_comparators(N)

# The RL code needs DECISIONS = length of the bit-string
DECISIONS = len(comparators)

def calc_score(state):
    """
    The RL loop will hand us a 0-1 vector of length DECISIONS.
    Each bit indicates whether we 'use' the corresponding comparator.

    Steps:
      1) Generate a random permutation of size N.
      2) For each comparator that is 'enabled' (bit == 1),
         compare-and-swap in that permutation.
      3) Return the length of the LIS of the resulting permutation.
    """
    used_comparators = [i for i, bit in enumerate(state[:DECISIONS]) if bit == 1]

    # Create a random permutation of [1..N]
    arr = list(range(1, N + 1))
    random.shuffle(arr)

    # Apply the chosen comparators
    for idx in used_comparators:
        i, j = comparators[idx]
        if arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]

    # Compute LIS
    return lengthOfLIS(arr)

if __name__ == "__main__":
    # Simple sanity check: see how many comparators we have
    print(f"For N={N}, we have {DECISIONS} comparators from Batcher OE network.")

    # Quick test: create a random 'state' bit-string and see the reward.
    test_state = np.random.choice([0, 1], size=DECISIONS)
    reward = calc_score(test_state)
    print(f"Random test bit-string => LIS reward = {reward}")
