import random

def hash_first_and_last_positions(h_coords, v_coords):
    positions = {}
    for i, val in enumerate(h_coords):
        if val not in positions:
            positions[val] = []
        positions[val].append(i)

    for i, val in enumerate(v_coords):
        if val not in positions:
            positions[val] = []
        positions[val].append(i)

    return positions

def generate_random_coords(n):
    base = list(range(1, n+1)) * 2
    
    h_coords = random.sample(base, len(base))
    v_coords = random.sample(base, len(base))
    
    return h_coords, v_coords

def find_intersections(positions_dict):
    rect_ids = list(positions_dict.keys())

    n = len(rect_ids)
    results = []
    score = 0

    for i in range(n):
        for j in range(i + 1, n):
            id1 = rect_ids[i]
            id2 = rect_ids[j]
            h0_1, h1_1, v0_1, v1_1 = positions_dict[id1]
            h0_2, h1_2, v0_2, v1_2 = positions_dict[id2]

            left1,  right1  = sorted([h0_1, h1_1])
            left2,  right2  = sorted([h0_2, h1_2])
            bot1,   top1    = sorted([v0_1, v1_1])
            bot2,   top2    = sorted([v0_2, v1_2])

            overlap_h = not (right1 < left2 or right2 < left1)
            overlap_v = not (top1   < bot2  or top2   < bot1)

            if overlap_h and overlap_v:
                results.append((id1, id2))
                score += 1

    return results, score

n = 10
h_coords_b, v_coords_b = generate_random_coords(n)
h_coords_r, v_coords_r = generate_random_coords(n)

positions_dict = hash_first_and_last_positions(h_coords, v_coords)

score = 0

intersections, score = find_intersections(positions_dict)
print("Intersecting pairs:", intersections)
print("Score:", score)