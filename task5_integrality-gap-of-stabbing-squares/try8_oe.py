import random
import time
import numpy as np
import pickle
from statistics import mean
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import networkx as nx
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

def build_traversal_graph(n):
    G = nx.DiGraph()
    left_nodes = [f'l{i}' for i in range(1, n+1)]
    right_nodes = [f'r{i}' for i in range(1, n+1)]
    for node in left_nodes + right_nodes:
        G.add_node(node)
    for i in range(1, n+1):
        G.add_edge(f'l{i}', f'r{i}')
    return G

def bit_traversal(G, bits, n=nRect):
    """
    Traverse the graph G using the bit-string (e.g. "11010") with the following rules:
    
      - Start at 'l1'.
      - At a node (say 'l{i}'), if there are two successors (['l{i+1}', 'r{i}']):
          * If the current bit is '1' and current != 'l{n-1}' then choose 'r{i}'.
          * But if current is 'l{n-1}', force the left successor.
          * If the bit is '0', choose the left successor.
          * Consume one bit per decision.
      - Continue until reaching 'l{n}'.
      - Finally, append any right nodes not visited.
    
    Returns the traversal path as a list of node labels.
    """
    path = ['l1']
    current = 'l1'
    bit_idx = 0
    visited_r = set()
    
    while current != f'l{n}':
        successors = list(G.successors(current))
        if len(successors) == 2 and bit_idx < len(bits):
            i_val = int(current[1:])
            if bits[bit_idx] == '1':
                if current == f'l{n-1}':
                    next_node = f'l{int(current[1:]) + 1}'
                else:
                    next_node = f'r{current[1:]}'
                    if next_node in successors:
                        path.append(next_node)
                        current = next_node
                        visited_r.add(current)
                        bit_idx += 1
                        continue
                    else:
                        next_node = f'l{int(current[1:]) + 1}'
            else:
                next_node = f'l{int(current[1:]) + 1}'
            bit_idx += 1
            if next_node in successors:
                path.append(next_node)
                current = next_node
                continue
        next_node = f'l{int(current[1:]) + 1}'
        path.append(next_node)
        current = next_node
    
    for i in range(1, n+1):
        r_node = f'r{i}'
        if r_node not in visited_r:
            path.append(r_node)
    
    return path

def decode_bit_sequence(bits, n):
    G = build_traversal_graph(n)
    if isinstance(bits, list):
        bits = ''.join(str(b) for b in bits)
    return bit_traversal(G, bits, n)

def order_to_intervals(order, n):
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
    order = decode_bit_sequence(seq_bits, n)
    intervals = order_to_intervals(order, n)
    return intervals

def run_gurobi(rectangles):
    x_coords = set()
    y_coords = set()
    for (L, R, T, B) in rectangles:
        if L not in x_coords:
            x_coords.add(L)
        if R not in x_coords:
            x_coords.add(R)
        if T not in y_coords:    
            y_coords.add(T)
        if B not in y_coords:
            y_coords.add(B)
    x_coords = sorted(list(x_coords))
    #print(x_coords)
    y_coords = sorted(list(y_coords))
    #print(y_coords)
    
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

model = Sequential()
model.add(Dense(FIRST_LAYER_NEURONS, activation="relu"))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.build((None, observation_space))
model.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate=LEARNING_RATE))
print(model.summary())

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
            # Split candidate into horizontal and vertical bits
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
    bits_str= super_actions[i].tolist()
    sc= super_rewards[i]
    print(f"bits={bits_str}, ratio={sc}")
avg= np.mean(super_rewards) if len(super_rewards)>0 else 0
print(f"\nAverage reward among final solutions= {avg}")
