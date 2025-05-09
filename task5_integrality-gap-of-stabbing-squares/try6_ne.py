import random
import time
import numpy as np
import pickle
from statistics import mean
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import networkx as nx  # no longer used for encoding
from gurobipy import Model, GRB, quicksum

nRect = 6
BITS_PER_DIM = 2 * nRect - 3 
TOTAL_BITS = 2 * BITS_PER_DIM 

LEARNING_RATE    = 0.0001
n_sessions       = 1000   
percentile       = 93     
super_percentile = 94     
FIRST_LAYER_NEURONS  = 128
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS  = 4
n_actions = 2  # binary actions

observation_space = 2 * TOTAL_BITS
len_game = TOTAL_BITS

##########################################################################
# New Encoding: Bitstring-based simulation of a Hamiltonian path.
##########################################################################
def simulate_path_from_bits(bits, n):
    """
    Simulate a valid permutation (Hamiltonian path) of 2*n nodes
    (l1, l2, …, ln, r1, r2, …, rn) based on a bitstring.
    The following conditions must hold:
      1. l_i appears before l_{i+1}
      2. r_i appears before r_{i+1}
      3. l_i appears before r_i
  
    Decision rule at each step:
      - If both a left move (l{next_left}) and a right move (r{next_right}) are legal,
        then if a bit is available:
          • If the bit is '1': pick the right move and consume the bit.
          • If the bit is '0': pick the left move and consume the bit.
      - If only one move is legal, force that move (without consuming a bit).
  
    Args:
        bits (str): A binary string (e.g. "11010") of maximum length 2*n-3.
        n (int): Number of left/right pairs.
  
    Returns:
        path (list): The resulting ordering of nodes.
    """
    path = ["l1"]
    next_left = 2   # Next left node to pick: l2, l3, ..., ln.
    next_right = 1  # Next right node available: r1, r2, ..., rn.
    bit_index = 0
    total_nodes = 2 * n

    while len(path) < total_nodes:
        legal_left = (next_left <= n)
        legal_right = (next_right < next_left and next_right <= n)
        
        if legal_left and legal_right:
            # Both moves legal: if a bit is available, decide accordingly.
            if bit_index < len(bits):
                b = bits[bit_index]
                bit_index += 1  # consume the bit
                if b == '1':
                    path.append(f"r{next_right}")
                    next_right += 1
                elif b == '0':
                    path.append(f"l{next_left}")
                    next_left += 1
                else:
                    raise ValueError("Bitstring must consist of '0' and '1' only.")
            else:
                # No bit left: default to left move.
                path.append(f"l{next_left}")
                next_left += 1
        elif legal_left:
            # Only left move is legal.
            path.append(f"l{next_left}")
            next_left += 1
        elif legal_right:
            # Only right move is legal.
            path.append(f"r{next_right}")
            next_right += 1
        else:
            break

    return path

def decode_bit_sequence(bits, n):
    """
    Decode a bit sequence into a permutation (Hamiltonian path)
    using the new bitstring-based simulation.
    """
    if isinstance(bits, list):
        bits = ''.join(str(b) for b in bits)
    return simulate_path_from_bits(bits, n)

def order_to_intervals(order, n):
    """
    Given a permutation (order) of nodes, assign coordinates to each node.
    Returns intervals [(L, R)] for rectangles 1..n where L is the coordinate
    of l{i} and R is the coordinate of r{i}.
    """
    coord = {}
    for idx, label in enumerate(order):
        coord[label] = idx + 1  # coordinates start at 1
    intervals = []
    for i in range(1, n+1):
        L = coord.get(f'l{i}', None)
        R = coord.get(f'r{i}', None)
        if L is None or R is None:
            raise ValueError(f"Missing coordinate for rectangle {i}")
        intervals.append((float(L), float(R)))
    return intervals

def get_intervals_from_bits(seq_bits, n):
    """
    Given a bit sequence (string or list) for one dimension, decode it to obtain the
    permutation and then map that permutation to intervals.
    """
    order = decode_bit_sequence(seq_bits, n)
    intervals = order_to_intervals(order, n)
    return intervals

##########################################################################
# Gurobi-based ILP/LP models for computing the integrality gap.
##########################################################################
def run_gurobi(rectangles):
    x_coords = set()
    y_coords = set()
    for (L, R, T, B) in rectangles:
        x_coords.add(L)
        x_coords.add(R)
        y_coords.add(T)
        y_coords.add(B)
    x_coords = sorted(list(x_coords))
    y_coords = sorted(list(y_coords))
    
    def x_candidates_for_rect(L, R):
        return [i for i, x in enumerate(x_coords) if L <= x <= R]
    def y_candidates_for_rect(T, B):
        return [j for j, y in enumerate(y_coords) if T <= y <= B]
    
    # ILP Model
    ILP_model = Model("stab_ILP")
    ILP_model.setParam("OutputFlag", 0)
    H = ILP_model.addVars(len(y_coords), vtype=GRB.BINARY)
    V = ILP_model.addVars(len(x_coords), vtype=GRB.BINARY)
    for (L, R, T, B) in rectangles:
        x_cands = x_candidates_for_rect(L, R)
        y_cands = y_candidates_for_rect(T, B)
        ILP_model.addConstr(quicksum(V[x] for x in x_cands) + quicksum(H[y] for y in y_cands) >= 1)
    ILP_model.setObjective(quicksum(V[i] for i in range(len(x_coords))) + quicksum(H[j] for j in range(len(y_coords))), GRB.MINIMIZE)
    ILP_model.optimize()
    ILP_value = ILP_model.objVal

    # LP Model
    LP_model = Model("stab_LP")
    LP_model.setParam("OutputFlag", 0)
    H_lp = LP_model.addVars(len(y_coords), vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
    V_lp = LP_model.addVars(len(x_coords), vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
    for (L, R, T, B) in rectangles:
        x_cands = x_candidates_for_rect(L, R)
        y_cands = y_candidates_for_rect(T, B)
        LP_model.addConstr(quicksum(V_lp[x] for x in x_cands) + quicksum(H_lp[y] for y in y_cands) >= 1)
    LP_model.setObjective(quicksum(V_lp[i] for i in range(len(x_coords))) + quicksum(H_lp[j] for j in range(len(y_coords))), GRB.MINIMIZE)
    LP_model.optimize()
    LP_value = LP_model.objVal

    return ILP_value, LP_value

def get_rectangles_from_bit_sequences(seq_h, seq_v):
    horiz_intervals = get_intervals_from_bits(seq_h, nRect)
    vert_intervals = get_intervals_from_bits(seq_v, nRect)
    rectangles = []
    for i in range(nRect):
        L, R = horiz_intervals[i]
        T, B = vert_intervals[i]
        rectangles.append((L, R, T, B))
    return rectangles

##########################################################################
# Build and compile the Keras model.
##########################################################################
model = Sequential()
model.add(Dense(FIRST_LAYER_NEURONS, activation="relu"))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.build((None, observation_space))
model.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate=LEARNING_RATE))
print(model.summary())

##########################################################################
# RL game-play functions.
##########################################################################
def play_game(n_sessions, actions, state_next, states, prob, step, total_score):
    """
    For each session, sample an action for the current step.
    When step == TOTAL_BITS, decode the candidate bit sequence and compute reward.
    """
    for i in range(n_sessions):
        action = 1 if random.random() < prob[i] else 0
        actions[i][step-1] = action
        state_next[i] = states[i, :, step-1].copy()
        if action > 0:
            state_next[i][step-1] = action
        state_next[i][TOTAL_BITS + step - 1] = 0
        if step < TOTAL_BITS:
            state_next[i][TOTAL_BITS + step] = 1
        terminal = (step == TOTAL_BITS)
        if terminal:
            candidate = state_next[i][:TOTAL_BITS]
            # Split candidate into horizontal and vertical bits.
            seq_h = candidate[:BITS_PER_DIM].tolist()
            seq_v = candidate[BITS_PER_DIM:TOTAL_BITS].tolist()
            seq_h_str = ''.join(str(bit) for bit in seq_h)
            seq_v_str = ''.join(str(bit) for bit in seq_v)
            rects = get_rectangles_from_bit_sequences(seq_h_str, seq_v_str)
            ilp, lp = run_gurobi(rects)
            total_score[i] = ilp / lp if lp > 0 else ilp
        else:
            states[i, :, step] = state_next[i]
    return actions, state_next, states, total_score, terminal

def generate_session(agent, n_sessions, verbose=0):
    """
    Generate n_sessions episodes using the RL agent.
    """
    states = np.zeros([n_sessions, observation_space, len_game], dtype=int)
    actions = np.zeros([n_sessions, len_game], dtype=int)
    state_next = np.zeros([n_sessions, observation_space], dtype=int)
    prob = np.zeros(n_sessions)
    states[:, TOTAL_BITS, 0] = 1  # initialize one-hot position indicator
    step = 0
    total_score = np.zeros(n_sessions)
    pred_time, play_time = 0, 0
    while True:
        step += 1
        t0 = time.time()
        prob = agent.predict(states[:, :, step-1], batch_size=n_sessions)
        pred_time += time.time() - t0
        t0 = time.time()
        actions, state_next, states, total_score, terminal = play_game(n_sessions, actions, state_next, states, prob, step, total_score)
        play_time += time.time() - t0
        if terminal:
            break
    if verbose:
        print(f"Predict time: {pred_time:.3f}, play time: {play_time:.3f}")
    return states, actions, total_score

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select sessions with rewards above the given percentile.
    """
    from numpy import percentile as np_percentile
    threshold = np_percentile(rewards_batch, percentile)
    elite_states = []
    elite_actions = []
    count = len(states_batch) * (100 - percentile) / 100.0
    for i in range(len(states_batch)):
        if rewards_batch[i] >= threshold - 1e-9:
            if count > 0 or rewards_batch[i] >= threshold + 1e-9:
                for st in states_batch[i]:
                    elite_states.append(st.tolist())
                for ac in actions_batch[i]:
                    elite_actions.append(ac)
                count -= 1
    return np.array(elite_states, dtype=int), np.array(elite_actions, dtype=int)

def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
    """
    Select sessions to survive to the next generation.
    """
    from numpy import percentile as np_percentile
    threshold = np_percentile(rewards_batch, percentile)
    super_states = []
    super_actions = []
    super_rewards = []
    count = len(states_batch) * (100 - percentile) / 100.0
    for i in range(len(states_batch)):
        if rewards_batch[i] >= threshold - 1e-9:
            if count > 0 or rewards_batch[i] >= threshold + 1e-9:
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
                count -= 1
    return np.array(super_states, dtype=int), np.array(super_actions, dtype=int), np.array(super_rewards, dtype=float)

##############################################################################
# Main Training Loop
##############################################################################
super_states = []  # now a list
super_actions = []
super_rewards = []
myRand = random.randint(0, 1000)

for iteration in range(1000000):
    t0 = time.time()
    sessions = generate_session(model, n_sessions, verbose=0)
    sess_time = time.time() - t0

    states_batch = np.array(sessions[0], dtype=int)  # shape: (n_sessions, observation_space, len_game)
    actions_batch = np.array(sessions[1], dtype=int)
    rewards_batch = np.array(sessions[2], dtype=float)
    states_batch = np.transpose(states_batch, (0, 2, 1))

    if len(super_states) > 0:
        states_batch = np.append(states_batch, np.array(super_states), axis=0)
        actions_batch = np.append(actions_batch, np.array(super_actions), axis=0)
        rewards_batch = np.append(rewards_batch, np.array(super_rewards))
    
    elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile)
    super_sess = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile)
    super_sess_list = [(super_sess[0][i], super_sess[1][i], super_sess[2][i]) for i in range(len(super_sess[2]))]
    super_sess_list.sort(key=lambda x: x[2], reverse=True)
    
    model.fit(elite_states, elite_actions, verbose=0)
    
    super_states = [x[0] for x in super_sess_list]
    super_actions = [x[1] for x in super_sess_list]
    super_rewards = [x[2] for x in super_sess_list]
    
    rewards_batch.sort()
    mean_all = np.mean(rewards_batch[-100:]) if len(rewards_batch) >= 100 else np.mean(rewards_batch)
    mean_best = np.mean(super_rewards) if len(super_rewards) > 0 else 0
    best_sorted = np.flip(np.sort(super_rewards))
    
    print(f"\nIteration {iteration}: Best rewards: {best_sorted}")
    print(f"Mean reward: {mean_all:.3f}, Session gen time: {sess_time:.3f}")
    
    if len(best_sorted) > 0 and best_sorted[0] >= 1.5:
        print(f"Reached integrality gap >= 1.5 at iteration {iteration}. Exiting.")
        break
    
    if iteration % 20 == 1:
        with open(f'best_species_pickle_{myRand}.txt', 'wb') as fp:
            pickle.dump(super_actions, fp)
        with open(f'best_species_txt_{myRand}.txt', 'w') as f:
            for item in super_actions:
                f.write(str(item) + "\n")
        with open(f'best_species_rewards_{myRand}.txt', 'w') as f:
            for item in super_rewards:
                f.write(str(item) + "\n")
        with open(f'best_100_rewards_{myRand}.txt', 'a') as f:
            f.write(str(mean_all) + "\n")
        with open(f'best_elite_rewards_{myRand}.txt', 'a') as f:
            f.write(str(mean_best) + "\n")
    
    if iteration % 200 == 2:
        with open(f'best_species_timeline_txt_{myRand}.txt', 'a') as f:
            if len(super_actions) > 0:
                f.write(str(super_actions[0]) + "\n")


print("\n======================")
print("Training loop ended.")
print("Final top solutions in descending reward order:\n")
for i in range(min(3, len(super_actions))):
    bits_str = super_actions[i].tolist()
    sc = super_rewards[i]
    print(f"bits={bits_str}, ratio={sc}")
avg = np.mean(super_rewards) if len(super_rewards) > 0 else 0
print(f"\nAverage reward among final solutions= {avg}")
