import itertools

def path_from_bits(bits, n):
    path = ["l1"]
    next_left = 2  
    next_right = 1 
    bit_index = 0
    total_nodes = 2 * n

    while len(path) < total_nodes:
        legal_left = (next_left <= n)
        legal_right = (next_right < next_left and next_right <= n)
        
        if legal_left and legal_right:
            if bit_index < len(bits):
                b = bits[bit_index]
                bit_index += 1 
                if b == '1':
                    path.append(f"r{next_right}")
                    next_right += 1
                elif b == '0':
                    path.append(f"l{next_left}")
                    next_left += 1
            else:
                path.append(f"l{next_left}")
                next_left += 1
        elif legal_left:
            path.append(f"l{next_left}")
            next_left += 1
        elif legal_right:
            path.append(f"r{next_right}")
            next_right += 1
        else:
            break

    return path

if __name__ == '__main__':
    n = 4
    max_bit_length = 2 * n - 3 
    all_bitstrings = [''.join(bits) for bits in itertools.product('01', repeat=max_bit_length)]
    
    final_arr = []

    print(f"permutations for {n}, len(bits) = {max_bit_length}:")
    for bitstring in all_bitstrings:
        path = path_from_bits(bitstring, n)
        print(f"bitstring: {bitstring} -> Path: {path}")
        arr = []
        for p in path:
            arr.append(int(p[-1]))
        final_arr.append(arr)

    for p in final_arr:
        print(p)
            

    
