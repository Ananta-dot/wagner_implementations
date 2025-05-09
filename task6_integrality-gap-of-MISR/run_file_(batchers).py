import random
import numpy as np
import pickle
import time
import math
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras import backend as K

def get_batcher_oe_comparators_py(n_):
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

n = 10
base_len = 2 * n  
comps = get_batcher_oe_comparators_py(base_len)
m = len(comps)               
DECISIONS = 2 * m            
observation_space = 2 * DECISIONS

LEARNING_RATE = 0.0001
n_sessions = 1000
percentile = 93
super_percentile = 94
FIRST_LAYER_NEURONS = 128
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

K.clear_session()
model = Sequential()
model.add(Dense(FIRST_LAYER_NEURONS, activation="relu", input_shape=(observation_space,)))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate=LEARNING_RATE))
print(model.summary())

def build_base_array(n_):
    arr = []
    for val in range(1, n_ + 1):
        arr.extend([val, val])
    return np.array(arr, dtype=int)

def apply_comps(baseA, bits, comps):
    arrc = baseA.copy()
    for i, bit in enumerate(bits):
        if bit == 1:
            a, b = comps[i]
            arrc[a], arrc[b] = arrc[b], arrc[a]
    return arrc

def solve_disjoint_rectangles(rectangles):
    n_rect = len(rectangles)
    
    min_x = min(rect[0][0] for rect in rectangles)
    max_x = max(rect[0][1] for rect in rectangles)
    min_y = min(rect[1][0] for rect in rectangles)
    max_y = max(rect[1][1] for rect in rectangles)
    
    x_points = list(range(min_x, max_x + 1))
    y_points = list(range(min_y, max_y + 1))
    candidate_points = [(x, y) for x in x_points for y in y_points]
    
    constraints = []
    for (x, y) in candidate_points:
        covering = []
        for i, rect in enumerate(rectangles):
            (x1, x2), (y1, y2) = rect
            if x1 < x < x2 and y1 < y < y2:
                covering.append(i)
        if len(covering) > 2:
            return 0.0, 0.0, 0.0  # case to not have more than 2 intersections, if then return 0
        elif len(covering) == 2:
            constraints.append(covering)

    # lp
    lp_val = 0.0
    try:
        model_lp = gp.Model("maxDisjointRectangles_LP")
        model_lp.setParam('OutputFlag', 0)
        x_lp = model_lp.addVars(n_rect, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")
        model_lp.setObjective(gp.quicksum(x_lp[i] for i in range(n_rect)), GRB.MAXIMIZE)
        for covers in constraints:
            model_lp.addConstr(gp.quicksum(x_lp[i] for i in covers) <= 1)
        model_lp.optimize()
        if model_lp.status == GRB.OPTIMAL:
            lp_val = model_lp.objVal
    except Exception as e:
        print("LP Exception:", e)
    
    # ilp
    ilp_val = 0.0
    try:
        model_ilp = gp.Model("maxDisjointRectangles_ILP")
        model_ilp.setParam('OutputFlag', 0)
        x_ilp = model_ilp.addVars(n_rect, vtype=GRB.BINARY, name="x")
        model_ilp.setObjective(gp.quicksum(x_ilp[i] for i in range(n_rect)), GRB.MAXIMIZE)
        for covers in constraints:
            model_ilp.addConstr(gp.quicksum(x_ilp[i] for i in covers) <= 1)
        model_ilp.optimize()
        if model_ilp.status == GRB.OPTIMAL:
            ilp_val = model_ilp.objVal
    except Exception as e:
        print("ILP Exception:", e)
    
    if ilp_val < 1e-9:
        ratio = 0
    else:
        ratio = lp_val / ilp_val
    return lp_val, ilp_val, ratio

def calc_score(state, n):
    s1 = state[:m]
    s2 = state[m:]
    base_arr = build_base_array(n)
    arrH = apply_comps(base_arr, s1, comps)
    arrV = apply_comps(base_arr, s2, comps)
    
    rectangles = []
    for i in range(1, n+1):
        x_indices = [idx for idx, val in enumerate(arrH) if val == i]
        y_indices = [idx for idx, val in enumerate(arrV) if val == i]
        if len(x_indices) < 2 or len(y_indices) < 2:
            rectangles.append(((0,0), (0,0)))
        else:
            x1, x2 = min(x_indices), max(x_indices)
            y1, y2 = min(y_indices), max(y_indices)
            rectangles.append(((x1, x2), (y1, y2)))
    
    lp_val, ilp_val, ratio = solve_disjoint_rectangles(rectangles)
    return ratio, arrH, arrV

def play_game(n_sess, actions, state_next, states, prob, step, total_score):
    for i in range(n_sess):
        action = 1 if random.random() < prob[i] else 0
        actions[i][step-1] = action
        state_next[i] = states[i, :, step-1].copy()
        if action > 0:
            state_next[i][step-1] = 1
        state_next[i][DECISIONS + step-1] = 0
        terminal = (step == DECISIONS)
        if not terminal:
            state_next[i][DECISIONS + step] = 1
        else:
            sc, arrH, arrV = calc_score(state_next[i][:DECISIONS], n)
            total_score[i] = sc
        if not terminal:
            states[i, :, step] = state_next[i].copy()
    return actions, state_next, states, total_score, terminal

def generate_session(agent, n_sess, verbose=1):
    obs_dim = 2 * DECISIONS  
    states = np.zeros((n_sess, obs_dim, DECISIONS), dtype=int)
    actions = np.zeros((n_sess, DECISIONS), dtype=int)
    state_next = np.zeros((n_sess, obs_dim), dtype=int)
    total_score = np.zeros(n_sess, dtype=float)
    
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
        actions, state_next, states, total_score, terminal = play_game(
            n_sess, actions, state_next, states, prob.reshape(-1), step, total_score
        )
        play_time += time.time() - t1
        
        if terminal:
            break
    if verbose:
        print(f"Predict: {pred_time:.3f}, play: {play_time:.3f}")
    return states, actions, total_score

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
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

super_states = np.empty((0, 2*DECISIONS, DECISIONS), dtype=int)
super_actions = np.empty((0, DECISIONS), dtype=int)
super_rewards = np.array([], dtype=float)

myRand = random.randint(0, 1000)
sessgen_time = 0.0
fit_time = 0.0
up_time = 0.0

TARGET_RATIO = 1.33

for iteration in range(1000000):
    t0 = time.time()
    states_batch, actions_batch, rewards_batch = generate_session(model, n_sessions, verbose=0)
    sessgen_time = time.time() - t0

    states_batch = np.concatenate([states_batch, super_states], axis=0)
    actions_batch = np.concatenate([actions_batch, super_actions], axis=0)
    rewards_batch = np.concatenate([rewards_batch, super_rewards])
    
    t1 = time.time()
    elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=50)
    sel_time1 = time.time() - t1
    
    t2 = time.time()
    s_states, s_actions, s_rewards = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90)
    sel_time2 = time.time() - t2
    
    t3 = time.time()
    tri = [(s_states[i], s_actions[i], s_rewards[i]) for i in range(len(s_rewards))]
    tri.sort(key=lambda x: x[2], reverse=True)
    sel_time3 = time.time() - t3
    
    t4 = time.time()
    if len(elite_states) > 0:
        model.fit(elite_states, elite_actions, verbose=0)
    fit_time = time.time() - t4
    
    t5 = time.time()
    super_states = np.array([x[0] for x in tri], dtype=int)
    super_actions = np.array([x[1] for x in tri], dtype=int)
    super_rewards = np.array([x[2] for x in tri], dtype=float)
    up_time = time.time() - t5
    
    rewards_batch.sort()
    mean_all_reward = np.mean(rewards_batch[-100:]) if len(rewards_batch) >= 100 else np.mean(rewards_batch)
    mean_best_reward = np.mean(super_rewards) if len(super_rewards) > 0 else 0.0
    
    print(f"\nIter {iteration}. Best individuals: {np.flip(np.sort(super_rewards))}")
    print(f"Mean reward (top100): {mean_all_reward:.3f}")
    print("Timing =>",
          f"sessgen: {sessgen_time:.3f},",
          f"select1: {sel_time1:.3f},",
          f"select2: {sel_time2:.3f},",
          f"select3: {sel_time3:.3f},",
          f"fit: {fit_time:.3f},",
          f"upd: {up_time:.3f}")
    
    if len(super_rewards) > 0 and max(super_rewards) >= TARGET_RATIO:
        print(f"Reached target ratio >= {TARGET_RATIO} at iteration= {iteration}. Breaking.")
        break
    
    if iteration % 20 == 1:
        with open(f'best_species_pickle_{myRand}.txt', 'wb') as fp:
            pickle.dump(super_actions, fp)

print("\n======================")
print("Training loop ended.")
print("Final top solutions in descending reward order:\n")
for i in range(1):
    bits_str = super_actions[i].tolist()
    sc = super_rewards[i]
    print(f"bits={bits_str}, ratio={sc}")

avg = np.mean(super_rewards) if len(super_rewards) > 0 else 0
print(f"\nAverage reward among final solutions= {avg}")
