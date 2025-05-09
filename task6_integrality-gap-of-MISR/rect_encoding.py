import math
import random
n = 14

bits_per_coord = (math.ceil(math.log2(n)))

def random_binary_string(n):
    return random.choices([0,1], k=n)

max_coord = 3 * n
DECISIONS = n * 4 * bits_per_coord

bits = random_binary_string(n*bits_per_coord*4)
coords = []

i = 0
while i < (len(bits)-bits_per_coord+1):
    s = ""
    for j in range(bits_per_coord):
        s += str(bits[i+j])
    coords.append(int(s,2))
    i += 4

print(coords)

print(math.ceil(math.log2(n)) ** 2)

