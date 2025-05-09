import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from statistics import mean

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import gurobipy as gp
from gurobipy import GRB

N = 6            # Number of squares
square_size = 2     # Each square is [x, x + square_size] x [y, y + square_size]

dummy_input_dim = 1
hidden_dim = 128

n_sessions      = 1000
percentile      = 93
super_percentile= 94
LEARNING_RATE   = 0.0001

def sinkhorn_tf(log_alpha, n_iters=20):
    # log_alpha: tensor of shape (batch, N, N)
    for _ in range(n_iters):
        log_alpha = log_alpha - tf.reduce_logsumexp(log_alpha, axis=2, keepdims=True)
        log_alpha = log_alpha - tf.reduce_logsumexp(log_alpha, axis=1, keepdims=True)
    return tf.exp(log_alpha)

K.clear_session()
inp = Input(shape=(dummy_input_dim,), name="input")
h = Dense(hidden_dim, activation='relu')(inp)
h = Dense(hidden_dim//2, activation='relu')(h)

# For each axis, output N*N logits.
logits_x = Dense(N * N, name="logits_x")(h)
logits_y = Dense(N * N, name="logits_y")(h)

# Reshape to (N, N)
logits_x_reshaped = Reshape((N, N), name="logits_x_reshaped")(logits_x)
logits_y_reshaped = Reshape((N, N), name="logits_y_reshaped")(logits_y)

# Apply Sinkhorn normalization via a Lambda layer.
perm_matrix_x = Lambda(lambda x: sinkhorn_tf(x, n_iters=20), name="sinkhorn_x")(logits_x_reshaped)
perm_matrix_y = Lambda(lambda x: sinkhorn_tf(x, n_iters=20), name="sinkhorn_y")(logits_y_reshaped)

# The network outputs two soft permutation matrices.
model = Model(inputs=inp, outputs=[perm_matrix_x, perm_matrix_y])
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE))
model.summary()

def solve_coverage_gurobi(squares, linesV, linesH):
    # ILP
    ilp = gp.Model("ExactILP")
    ilp.setParam('OutputFlag', 0)
    varV_ilp = {}
    for v in linesV:
        varV_ilp[v] = ilp.addVar(vtype=GRB.BINARY, name=f"V_{v}")
    varH_ilp = {}
    for h in linesH:
        varH_ilp[h] = ilp.addVar(vtype=GRB.BINARY, name=f"H_{h}")
    ilp.update()
    for (xl, xr, yb, yt) in squares:
        cExpr = gp.LinExpr()
        for v in linesV:
            if v >= xl and v <= xr:
                cExpr.addTerms(1.0, varV_ilp[v])
        for h in linesH:
            if h >= yb and h <= yt:
                cExpr.addTerms(1.0, varH_ilp[h])
        ilp.addConstr(cExpr >= 1)
    ilp.setObjective(
        gp.quicksum(varV_ilp[v] for v in linesV) + gp.quicksum(varH_ilp[h] for h in linesH),
        GRB.MINIMIZE
    )
    ilp.optimize()
    if ilp.status == GRB.OPTIMAL:
        ILP_val = ilp.objVal
    else:
        ILP_val = None
    ilp.dispose()

    # LP
    lp = gp.Model("ExactLP")
    lp.setParam('OutputFlag', 0)
    varV_lp = {}
    for v in linesV:
        varV_lp[v] = lp.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"V_{v}")
    varH_lp = {}
    for h in linesH:
        varH_lp[h] = lp.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"H_{h}")
    lp.update()
    for (xl, xr, yb, yt) in squares:
        cExpr = gp.LinExpr()
        for v in linesV:
            if v >= xl and v <= xr:
                cExpr.addTerms(1.0, varV_lp[v])
        for h in linesH:
            if h >= yb and h <= yt:
                cExpr.addTerms(1.0, varH_lp[h])
        lp.addConstr(cExpr >= 1)
    lp.setObjective(
        gp.quicksum(varV_lp[v] for v in linesV) + gp.quicksum(varH_lp[h] for h in linesH),
        GRB.MINIMIZE
    )
    lp.optimize()
    if lp.status == GRB.OPTIMAL:
        LP_val = lp.objVal
    else:
        LP_val = None
    lp.dispose()

    return ILP_val, LP_val

###############################################################################
# Key Change: We randomly sample each row of the permutation matrix 
# instead of using np.argmax.
###############################################################################
def calc_score(perm_x, perm_y):
    """
    perm_x and perm_y are (N, N) soft permutations from Sinkhorn.
    We sample an index from each row's probability distribution,
    so we get a random permutation each time.
    """
    # Sample from each row i in perm_x
    perm_indices_x = []
    for row in perm_x:
        row = row / row.sum()  # just to ensure numerical stability
        idx = np.random.choice(N, p=row)
        perm_indices_x.append(idx)
    perm_indices_x = np.array(perm_indices_x)
    
    # Sample from each row i in perm_y
    perm_indices_y = []
    for row in perm_y:
        row = row / row.sum()
        idx = np.random.choice(N, p=row)
        perm_indices_y.append(idx)
    perm_indices_y = np.array(perm_indices_y)

    # Define base coordinate arrays.
    baseX = np.arange(1, N * square_size + 1, square_size)  # e.g. [1,3,5,7,9,11]
    baseY = np.arange(1, N * square_size + 1, square_size)
    
    # Reorder using the sampled permutation indices.
    arrX = [baseX[i] for i in perm_indices_x]
    arrY = [baseY[i] for i in perm_indices_y]
    
    squares = []
    for i in range(N):
        xl = arrX[i]
        xr = xl + square_size
        yl = arrY[i]
        yr = yl + square_size
        squares.append((xl, xr, yl, yr))
    
    linesV_set = set()
    linesH_set = set()
    for (xl, xr, yl, yr) in squares:
        linesV_set.add(xl)
        linesV_set.add(xr)
        linesH_set.add(yl)
        linesH_set.add(yr)
    linesV = sorted(list(linesV_set))
    linesH = sorted(list(linesH_set))
    
    ILP_val, LP_val = solve_coverage_gurobi(squares, linesV, linesH)
    if ILP_val is None or LP_val is None or abs(LP_val) < 1e-9:
        return 0.0
    return float(ILP_val)/ float(LP_val)

def play_game(n_sessions, states, total_score):
    dummy_input = np.ones((n_sessions, dummy_input_dim), dtype=np.float32)
    preds = model.predict(dummy_input, batch_size=n_sessions)
    perm_matrices_x = preds[0]  # shape: (n_sessions, N, N)
    perm_matrices_y = preds[1]  # shape: (n_sessions, N, N)
    for i in range(n_sessions):
        score = calc_score(perm_matrices_x[i], perm_matrices_y[i])
        total_score[i] = score
        # Just for logging, store a dummy action
        states[i, 0] = score  # e.g., store the score itself
    return states, total_score

def generate_session(n_sess, verbose=1):
    states = np.zeros((n_sess, 1), dtype=np.float32)
    total_score = np.zeros(n_sess)
    t0 = time.time()
    states, total_score = play_game(n_sess, states, total_score)
    elapsed = time.time() - t0
    if verbose:
        print(f"Predict & evaluation time: {elapsed:.3f} sec")
    return states, total_score

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    counter = n_sessions * (100.0 - percentile) / 100.0
    thresh = np.percentile(rewards_batch, percentile)
    elite_states = []
    elite_actions = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= thresh - 1e-9:
            if counter > 0 or (rewards_batch[i] > thresh + 1e-9):
                elite_states.append(states_batch[i])
                elite_actions.append(actions_batch[i])
            counter -= 1
    return np.array(elite_states, dtype=np.float32), np.array(elite_actions, dtype=np.float32)

def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
    counter = n_sessions * (100.0 - percentile) / 100.0
    thresh = np.percentile(rewards_batch, percentile)
    super_states = []
    super_actions = []
    super_rewards = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= thresh - 1e-9:
            if counter > 0 or (rewards_batch[i] > thresh + 1e-9):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
            counter -= 1
    return (np.array(super_states, dtype=np.float32), 
            np.array(super_actions, dtype=np.float32),
            np.array(super_rewards, dtype=np.float32))

super_states = np.empty((0, 1), dtype=np.float32)
super_actions = np.empty((0, 1), dtype=np.float32)
super_rewards = np.array([])

myRand = random.randint(0, 1000)

for iteration in range(1000000):
    t0 = time.time()
    sessions, rewards_batch = generate_session(n_sessions, verbose=0)
    sessgen_time = time.time() - t0

    actions_batch = sessions.copy()
    states_batch = sessions.copy()

    states_batch = np.append(states_batch, super_states, axis=0)
    actions_batch = np.append(actions_batch, super_actions, axis=0)
    rewards_batch = np.append(rewards_batch, super_rewards)

    t1 = time.time()
    elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)
    sel_time1 = time.time() - t1

    t2 = time.time()
    s_states, s_actions, s_rewards = select_super_sessions(states_batch, actions_batch, rewards_batch, super_percentile)
    sel_time2 = time.time() - t2

    t3 = time.time()
    tri = [(s_states[i], s_actions[i], s_rewards[i]) for i in range(len(s_rewards))]
    tri.sort(key=lambda x: x[2], reverse=True)
    sel_time3 = time.time() - t3

    t4 = time.time()
    if len(elite_states) > 0:
        # For training, we use a dummy target with shape (batch, N, N).
        dummy_target = np.tile(np.eye(N, dtype=np.float32), (elite_states.shape[0], 1, 1))
        # Fit the network on both x and y branches.
        model.fit(elite_states, [dummy_target, dummy_target], verbose=0)
    fit_time = time.time() - t4

    t5 = time.time()
    super_states = np.array([x[0] for x in tri], dtype=np.float32).reshape(-1, 1)
    super_actions = np.array([x[1] for x in tri], dtype=np.float32).reshape(-1, 1)
    super_rewards = np.array([x[2] for x in tri], dtype=np.float32)
    up_time = time.time() - t5

    rewards_batch.sort()
    if len(rewards_batch) >= 100:
        mean_all_reward = np.mean(rewards_batch[-100:])
    else:
        mean_all_reward = np.mean(rewards_batch)
    mean_best_reward = np.mean(super_rewards) if len(super_rewards) > 0 else 0.0

    print(f"\nIter {iteration}. Best individuals: {np.flip(np.sort(super_rewards))}")
    print(f"Mean reward: {mean_all_reward:.3f}")
    print("Timing =>"
          + f" sessgen: {sessgen_time:.3f},"
          + f" select1: {sel_time1:.3f},"
          + f" select2: {sel_time2:.3f},"
          + f" select3: {sel_time3:.3f},"
          + f" fit: {fit_time:.3f},"
          + f" upd: {up_time:.3f}"
    )

    if len(super_rewards) > 0 and max(super_rewards) >= 1.5:
        print(f"Reached ratio >=1.5 at iteration= {iteration}. Breaking.")
        break

print("\n======================")
print("Training loop ended.")
print("Final top solutions in descending reward order:\n")
for i in range(min(3, len(super_actions))):
    solution = super_actions[i].tolist()
    sc = super_rewards[i]
    print(f"solution={solution}, ratio={sc}")
avg = np.mean(super_rewards) if len(super_rewards) > 0 else 0
print(f"\nAverage reward among final solutions= {avg}")

dummy_input = np.ones((1, dummy_input_dim), dtype=np.float32)
soft_perm_matrix_x, soft_perm_matrix_y = model.predict(dummy_input)

# Convert the soft permutation matrices (shape: (1, N, N)) to hard permutations:
perm_indices_x = np.argmax(soft_perm_matrix_x[0], axis=1)  # shape: (N,)
perm_indices_y = np.argmax(soft_perm_matrix_y[0], axis=1)  # shape: (N,)

# Print permutation indices:
print("\nBest Permutation Indices:")
print("X-axis indices:", perm_indices_x)
print("Y-axis indices:", perm_indices_y)

# Define base coordinate arrays.
baseX = np.arange(1, N * square_size + 1, square_size)  # e.g., [1, 3, 5, 7, 9, 11]
baseY = np.arange(1, N * square_size + 1, square_size)

# Reorder the base arrays using the extracted permutation indices.
arrX = [baseX[i] for i in perm_indices_x]
arrY = [baseY[i] for i in perm_indices_y]

print("Reordered X coordinates:", arrX)
print("Reordered Y coordinates:", arrY)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots()
for i in range(N):
    rect = patches.Rectangle((arrX[i], arrY[i]), square_size, square_size,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(arrX[i] + square_size/2, arrY[i] + square_size/2, f"{i}",
            color='blue', fontsize=12, ha='center', va='center')

ax.set_xlim(0, N * square_size + 2)
ax.set_ylim(0, N * square_size + 2)
ax.set_aspect('equal', adjustable='box')
plt.title("Best Solution: Square Placements")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()