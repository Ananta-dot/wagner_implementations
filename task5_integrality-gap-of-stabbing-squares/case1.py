from tensorflow.keras import Input, Model
from tensorflow.keras import layers, initializers
import random
import numpy as np
import pickle
import time
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Layer, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import initializers
import gurobipy as gp
from gurobipy import GRB
from numba import njit

# Optionally use numba to jit some inner routines.
from numba import njit

#########################
# Utility functions
#########################

def get_batcher_oe_comparators_py(n_):
    """
    Returns a list of comparator pairs for Batcher’s odd-even comparator network
    on an array of length n_. (See e.g. [1])
    """
    comps = []
    p = 1
    while p < n_:
        k = p
        while k >= 1:
            j_start = k % p
            j_end = n_ - 1 - k
            step_j = 2 * k
            j = j_start
            while j <= j_end:
                i_max = min(k - 1, n_ - j - k - 1)
                for i in range(i_max + 1):
                    left = (i + j) // (2 * p)
                    right = (i + j + k) // (2 * p)
                    if left == right:
                        comps.append((i + j, i + j + k))
                j += step_j
            k //= 2
        p *= 2
    return comps

n = 9  # Number of distinct labels/rectangles; can be seen as the problem “size”
base_len = 2 * n  
comps = get_batcher_oe_comparators_py(base_len)
m = len(comps)               # number of comparators
DECISIONS = 2 * m            # we will make decisions for both horizontal and vertical orders
observation_space = 2 * DECISIONS

# RL/learning parameters:
LEARNING_RATE = 0.0001  
n_sessions = 1000  
percentile = 93  
super_percentile = 94  
FIRST_LAYER_NEURONS = 256    # increased size for improved policy
SECOND_LAYER_NEURONS = 128
THIRD_LAYER_NEURONS = 32     # larger hidden representation

# Target integrality gap (the theoretical optimum for our instance is 1.5)
TARGET_GAP = 1.3

# A simple attention layer for 1D inputs; if your input is 1D, it may not provide dramatic gains.
from keras.layers import Layer

class AttentionLayer(layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                initializer=initializers.RandomNormal(),
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W)) 
        alpha = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * alpha, axis=1, keepdims=True)  # Keep 2D shape

def build_base_array(n_):
    arr = []
    for val in range(1, n_+1):
        arr.extend([val, val])
    return np.array(arr, dtype=int)

def apply_comps(baseA, bits, comps):
    """
    Given a base ordering and a binary sequence (for comparators), apply swaps for each comparator where bit==1.
    This returns a permutation array.
    """
    arrc = baseA.copy()
    for i, bit in enumerate(bits):
        if bit == 1:
            a, b = comps[i]
            arrc[a], arrc[b] = arrc[b], arrc[a]
    return arrc

def rect_intersect(r1, r2):
    """
    Check for “open” intersection of two rectangles.
    Each rectangle is represented as ((x1, x2), (y1, y2)).
    We require strict inequality for an overlap.
    """
    (x1a, x2a), (y1a, y2a) = r1
    (x1b, x2b), (y1b, y2b) = r2
    # Open intersection: boundaries do not count.
    return (x1a < x2b and x1b < x2a and y1a < y2b and y1b < y2a)

def solve_disjoint_rectangles_iterative(rectangles, conflict_threshold=5):
    """
    Iterative removal procedure inspired by [1]:
      - Build a conflict graph (nodes = rectangle indices; an edge exists if rectangles intersect).
      - Iteratively remove nodes whose degree exceeds the conflict_threshold.
      - Then run a simple greedy independent set algorithm.
      
    Returns: (lp_val, ilp_val, gap_ratio)
      where lp_val is defined as total fractional weight (each rectangle worth 1 unit fractionally);
            ilp_val is size of the independent set found.
    """
    n_rect = len(rectangles)
    if n_rect == 0:
        return 0.0, 0.0, 0.0

    # Build conflict graph (as a dictionary with node => set(neighbor indices))
    nodes = list(range(n_rect))
    conflict_graph = { i: set() for i in nodes }
    for i in range(n_rect):
        for j in range(i+1, n_rect):
            if rect_intersect(rectangles[i], rectangles[j]):
                conflict_graph[i].add(j)
                conflict_graph[j].add(i)

    # Iterative removal: while some node in current set R has degree > conflict_threshold, remove the highest-degree node.
    R = set(nodes)
    while True:
        candidate = None
        max_degree = -1
        for i in list(R):
            deg = len(conflict_graph[i])
            if deg > max_degree:
                max_degree = deg
                candidate = i
        if max_degree > conflict_threshold:
            R.remove(candidate)
            # Remove candidate from its neighbors' sets:
            for j in conflict_graph[candidate]:
                if j in R:
                    conflict_graph[j].discard(candidate)
            # Optionally remove the candidate entirely.
            del conflict_graph[candidate]
        else:
            break

    # Now use a simple greedy method on remaining nodes in R.
    independent_set = []
    taken = set()
    # Sort nodes in R by increasing degree (heuristic)
    sorted_nodes = sorted(R, key=lambda i: len(conflict_graph[i]))
    for i in sorted_nodes:
        if i not in taken:
            independent_set.append(i)
            # Remove (mark) all neighbors of i.
            for neighbor in conflict_graph[i]:
                taken.add(neighbor)

    # We define lp_val as the total number of original rectangles (each of weight 1)
    lp_val = float(n_rect)
    ilp_val = float(len(independent_set))
    gap = lp_val / ilp_val if ilp_val > 0 else 0.0
    return lp_val, ilp_val, gap

def solve_disjoint_rectangles(rectangles):
    return solve_disjoint_rectangles_iterative(rectangles, conflict_threshold=5)

def calc_score(state, n):
    """
    Given a state (binary vector of length DECISIONS representing decisions for horizontal comparisons followed by vertical)
    we:  
      1. Build the horizontal permutation using the first m bits
      2. Build the vertical permutation using the next m bits
      3. For each label i (1 <= i <= n), find its two indices in each array to define a rectangle
      4. Compute the integrality gap using our iterative disjoint–rectangle solver.
      
    A modified reward function is used: we define reward = TARGET_GAP - abs(TARGET_GAP - gap)
    so that the best reward is achieved when gap == TARGET_GAP.
    """
    s1 = state[:m]
    s2 = state[m:]
    base_arr = build_base_array(n)
    arrH = apply_comps(base_arr, s1, comps)
    arrV = apply_comps(base_arr, s2, comps)
    
    rectangles = []
    # For each label, find the minimal and maximal indices in the horizontal and vertical arrays.
    for i in range(1, n+1):
        x_indices = [idx for idx, val in enumerate(arrH) if val == i]
        y_indices = [idx for idx, val in enumerate(arrV) if val == i]
        if len(x_indices) < 2 or len(y_indices) < 2:
            # Use a dummy rectangle if insufficient occurrences.
            rectangles.append(((0, 0), (0, 0)))
        else:
            x1, x2 = min(x_indices), max(x_indices)
            y1, y2 = min(y_indices), max(y_indices)
            rectangles.append(((x1, x2), (y1, y2)))
    
    lp_val, ilp_val, gap = solve_disjoint_rectangles(rectangles)
    # Gap should ideally approach TARGET_GAP (1.5). We use a reward function that gives maximum
    # reward when gap is exactly TARGET_GAP.
    reward = TARGET_GAP - abs(TARGET_GAP - gap)
    # Optionally add an exploration bonus (e.g. a small factor times the diversity of arrH/arrV)
    return reward, arrH, arrV

#########################
# RL game and session generation
#########################

def play_game(n_sess, actions, state_next, states, prob, step, total_score):
    """
    For every session, sample the action according to probability provided by the neural network,
    update the state by propagating the previous state and setting the current decision.
    """
    for i in range(n_sess):
        # sample action from probability distribution: here simple Bernoulli.
        action = 1 if random.random() < prob[i] else 0
        actions[i][step-1] = action
        # Propagate previous state.
        state_next[i] = states[i, :, step-1].copy()
        # Set decision for the current step.
        if action > 0:
            state_next[i][step-1] = 1
        else:
            state_next[i][step-1] = 0
        # Mark the next decision index active if available.
        state_next[i][DECISIONS + step-1] = 0
        terminal = (step == DECISIONS)
        if not terminal:
            state_next[i][DECISIONS + step] = 1
        else:
            # When session ends, compute the reward.
            sc, _, _ = calc_score(state_next[i][:DECISIONS], n)
            total_score[i] = sc
        if not terminal:
            states[i, :, step] = state_next[i].copy()
    return actions, state_next, states, total_score, terminal

jitted_play_game = njit()(play_game)  # Use numba jitting if available

def generate_session(agent, n_sess, iteration=0, verbose=1):
    """
    Generate n_sess sessions of decisions.
    (Optionally use curriculum learning: e.g. for early iterations use a smaller n.)
    The state is represented by a (observation_space x len_game) integer matrix.
    """
    len_game = DECISIONS
    obs_dim = observation_space
    # Optionally: Use curriculum learning by scaling n for early iterations.
    curr_n = n  # set effective n; for example, you might set curr_n = max(4, int(n*0.8)) if iteration < 50

    states = np.zeros((n_sess, obs_dim, len_game), dtype=int)
    actions = np.zeros((n_sess, len_game), dtype=int)
    state_next = np.zeros((n_sess, obs_dim), dtype=int)
    total_score = np.zeros(n_sess, dtype=float)
    
    # Initial state: mark first k decision bit active.
    states[:, DECISIONS, 0] = 1
    step = 0
    pred_time = 0.0
    play_time = 0.0
    while True:
        step += 1
        t0 = time.time()
        prob = agent.predict(states[:, :, step-1], batch_size=n_sess)
        pred_time += time.time() - t0
        
        t1 = time.time()
        actions, state_next, states, total_score, terminal = play_game(n_sess, actions, state_next, states, prob.reshape(-1), step, total_score)
        play_time += time.time() - t1
        
        if terminal:
            break
    if verbose:
        print(f"Generate-session: predict {pred_time:.3f}s, play {play_time:.3f}s")
    return states, actions, total_score

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select the sessions translating to steps (states and actions) with reward above the given percentile.
    """
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_states = []
    elite_actions = []
    needed = len(states_batch) * (100.0 - percentile) / 100.0
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold - 1e-9:
            if (needed > 0) or (rewards_batch[i] > reward_threshold + 1e-9):
                for t in range(DECISIONS):
                    elite_states.append(states_batch[i][:, t])
                    elite_actions.append(actions_batch[i][t])
            needed -= 1
    elite_states = np.array(elite_states, dtype=int)
    elite_actions = np.array(elite_actions, dtype=int)
    return elite_states, elite_actions

def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
    """
    Select the “super” sessions that will survive to the next iteration.
    """
    reward_threshold = np.percentile(rewards_batch, percentile)
    super_states = []
    super_actions = []
    super_rewards = []
    needed = len(states_batch) * (100.0 - percentile) / 100.0
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold - 1e-9:
            if (needed > 0) or (rewards_batch[i] > reward_threshold + 1e-9):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
            needed -= 1
    return (np.array(super_states, dtype=int),
            np.array(super_actions, dtype=int),
            np.array(super_rewards, dtype=float))

K.clear_session()

model = Sequential([
    Input(shape=(observation_space,)),  # Explicit input shape
    Dense(FIRST_LAYER_NEURONS, activation='relu'),
    Dense(SECOND_LAYER_NEURONS, activation='relu'),
    AttentionLayer(),
    Dense(THIRD_LAYER_NEURONS, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate=LEARNING_RATE))
print(model.summary())

#########################
# Main training loop
#########################
super_states = np.empty((0, observation_space, DECISIONS), dtype=int)
super_actions = np.empty((0, DECISIONS), dtype=int)
super_rewards = np.array([], dtype=float)

myRand = random.randint(0, 1000)
sessgen_time = 0.0
fit_time = 0.0
upd_time = 0.0

for iteration in range(1000000):
    t0 = time.time()
    states_batch, actions_batch, rewards_batch = generate_session(model, n_sessions, iteration=iteration, verbose=0)
    sessgen_time = time.time() - t0
    
    # Concatenate with surviving sessions from previous iterations (curriculum survival)
    if super_states.shape[0] > 0:
        states_batch = np.concatenate([states_batch, super_states], axis=0)
        actions_batch = np.concatenate([actions_batch, super_actions], axis=0)
        rewards_batch = np.concatenate([rewards_batch, super_rewards])
    
    t1 = time.time()
    elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile)
    sel_time1 = time.time() - t1
    
    t2 = time.time()
    s_states, s_actions, s_rewards = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile)
    sel_time2 = time.time() - t2
    
    # Sort super sessions by reward
    t3 = time.time()
    super_list = [(s_states[i], s_actions[i], s_rewards[i]) for i in range(len(s_rewards))]
    super_list.sort(key=lambda x: x[2], reverse=True)
    sel_time3 = time.time() - t3
    
    t4 = time.time()
    if elite_states.shape[0] > 0:
        model.fit(elite_states, elite_actions, verbose=0)
    fit_time = time.time() - t4
    
    t5 = time.time()
    super_states = np.array([x[0] for x in super_list], dtype=int)
    super_actions = np.array([x[1] for x in super_list], dtype=int)
    super_rewards = np.array([x[2] for x in super_list], dtype=float)
    upd_time = time.time() - t5
    
    rewards_sorted = np.sort(rewards_batch)
    mean_all_reward = np.mean(rewards_sorted[-100:]) if len(rewards_sorted) >= 100 else np.mean(rewards_sorted)
    mean_best_reward = np.mean(super_rewards) if len(super_rewards) > 0 else 0.0
    
    print(f"\nIteration {iteration}. Best super-rewards: {np.flip(np.sort(super_rewards))}")
    print(f"Mean top100 reward: {mean_all_reward:.3f}")
    print("Timings -> sessgen: {:.3f}, select1: {:.3f}, select2: {:.3f}, select3: {:.3f}, fit: {:.3f}, update: {:.3f}"
          .format(sessgen_time, sel_time1, sel_time2, sel_time3, fit_time, upd_time))
    
    # If we have found a construction with reward nearly TARGET_GAP, stop.
    if len(super_rewards) > 0 and max(super_rewards) >= TARGET_GAP - 1e-4:
        print(f"Reached target gap (reward ~ {TARGET_GAP}) at iteration {iteration}.")
        break
    
    # Optionally, every 20 iterations, save the best species.
    if iteration % 20 == 1:
        with open(f'best_species_pickle_{myRand}.txt', 'wb') as fp:
            pickle.dump(super_actions, fp)

print("\n======================")
print("Training loop ended.")
print("Final top solutions (in descending reward order):")
for i in range(min(5, len(super_rewards))):
    bits_str = super_actions[i].tolist()
    sc = super_rewards[i]
    print(f"bits={bits_str}, reward (approx gap)={sc}")
avg = np.mean(super_rewards) if len(super_rewards)>0 else 0
print(f"\nAverage reward among final solutions = {avg:.3f}")
