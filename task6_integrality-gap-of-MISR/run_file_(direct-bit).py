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
from keras.optimizers import SGD
from keras import backend as K

num_rectangles = 14
bits_per_coord = math.ceil(math.log2(num_rectangles))
DECISIONS = num_rectangles * 4 * bits_per_coord
observation_space = 2 * DECISIONS

nearest_log = math.ceil(math.log2(num_rectangles)) ** 2

coord_max = (nearest_log + 1) * (nearest_log + 1)

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


def decode_rectangles_xywh_from_bits(
    bit_array, num_rectangles, bits_per_coord, coord_max
):
    """
    Decodes a bit array into a list of rectangles, each rectangle given by
      ( (x_left, x_right), (y_bottom, y_top) )
    using the (x, y, w, h) approach:
      x2 = x + w,   y2 = y + h,
    then clamp if x2 > coord_max or y2 > coord_max.

    We have 4 groups of bits_per_coord bits for each rectangle:
      1) x bits
      2) y bits
      3) w bits
      4) h bits

    All are scaled to [0..coord_max]. No fraction-of-size clamp is used.
    """
    bits_per_rect = 4 * bits_per_coord
    expected_len = num_rectangles * bits_per_rect
    if len(bit_array) != expected_len:
        raise ValueError(f"Bit array length mismatch: want={expected_len}, got={len(bit_array)}")

    max_int = (1 << bits_per_coord) - 1  # e.g. if bits_per_coord=6 => max_int=63
    rectangles = []
    idx = 0
    for r in range(num_rectangles):
        # x bits
        x_val = int("".join(str(b) for b in bit_array[idx: idx + bits_per_coord]), 2)
        idx += bits_per_coord
        # y bits
        y_val = int("".join(str(b) for b in bit_array[idx: idx + bits_per_coord]), 2)
        idx += bits_per_coord
        # w bits
        w_val = int("".join(str(b) for b in bit_array[idx: idx + bits_per_coord]), 2)
        idx += bits_per_coord
        # h bits
        h_val = int("".join(str(b) for b in bit_array[idx: idx + bits_per_coord]), 2)
        idx += bits_per_coord

        # Scale to [0..coord_max]
        x = (x_val / max_int) * coord_max if max_int > 0 else 0
        y = (y_val / max_int) * coord_max if max_int > 0 else 0
        w = (w_val / max_int) * coord_max if max_int > 0 else 0
        h = (h_val / max_int) * coord_max if max_int > 0 else 0

        # Build final corners
        x_left = int(x)
        x_right = int(x + w)
        y_bottom = int(y)
        y_top = int(y + h)

        # clamp to [0..coord_max]
        if x_right > coord_max:
            x_right = coord_max
        if y_top > coord_max:
            y_top = coord_max

        rectangles.append(((x_left, x_right), (y_bottom, y_top)))
    return rectangles


def plot_rectangles(rectangles):
    fig, ax = plt.subplots(figsize=(8, 8))
    xs, ys = [], []
    for idx, ((x_left, x_right), (y_bottom, y_top)) in enumerate(rectangles):
        w = x_right - x_left
        h = y_top - y_bottom
        color = (random.random(), random.random(), random.random(), 0.5)
        rect = plt.Rectangle(
            (x_left, y_bottom),
            w,
            h,
            edgecolor="black",
            facecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)
        cx, cy = x_left + w / 2, y_bottom + h / 2
        ax.text(cx, cy, f"{idx+1}", fontsize=12, ha="center", va="center", color="black")
        xs.extend([x_left, x_right])
        ys.extend([y_bottom, y_top])
    margin = 5
    if xs and ys:
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Decoded Rectangles: (x,y,w,h) => corners")
    plt.grid(True)
    plt.show()


def solve_disjoint_rectangles(rectangles):
    """
    Coverage-based approach:
      - For each integer point in bounding box, sum of chosen rects covering that point <= 1
      - We solve LP, then ILP.

    Returns (lp_val, ilp_val, ratio).
    """
    n_rect = len(rectangles)
    if n_rect == 0:
        return 0, 0, 0

    min_x = min(r[0][0] for r in rectangles)
    max_x = max(r[0][1] for r in rectangles)
    min_y = min(r[1][0] for r in rectangles)
    max_y = max(r[1][1] for r in rectangles)

    x_points = list(range(min_x, max_x + 1))
    y_points = list(range(min_y, max_y + 1))
    candidate_points = [(x, y) for x in x_points for y in y_points]

    constraints = []
    for (x, y) in candidate_points:
        covering = []
        for i, ((x1, x2), (y1, y2)) in enumerate(rectangles):
            if x1 < x < x2 and y1 < y < y2:
                covering.append(i)
        if len(covering) > 1:
            constraints.append(covering)

    lp_val = 0.0
    ilp_val = 0.0
    # Solve LP
    try:
        model_lp = gp.Model("RectLP")
        model_lp.Params.OutputFlag = 0
        x_lp = model_lp.addVars(n_rect, lb=0, ub=1, vtype=GRB.CONTINUOUS)
        model_lp.setObjective(gp.quicksum(x_lp[i] for i in range(n_rect)), GRB.MAXIMIZE)
        for covers in constraints:
            model_lp.addConstr(gp.quicksum(x_lp[i] for i in covers) <= 1)
        model_lp.optimize()
        if model_lp.status == GRB.OPTIMAL:
            lp_val = model_lp.objVal
    except Exception as e:
        print("LP Exception:", e)

    # Solve ILP
    try:
        model_ilp = gp.Model("RectILP")
        model_ilp.Params.OutputFlag = 0
        x_ilp = model_ilp.addVars(n_rect, vtype=GRB.BINARY)
        model_ilp.setObjective(gp.quicksum(x_ilp[i] for i in range(n_rect)), GRB.MAXIMIZE)
        for covers in constraints:
            model_ilp.addConstr(gp.quicksum(x_ilp[i] for i in covers) <= 1)
        model_ilp.optimize()
        if model_ilp.status == GRB.OPTIMAL:
            ilp_val = model_ilp.objVal
    except Exception as e:
        print("ILP Exception:", e)

    ratio = lp_val / ilp_val if ilp_val > 1e-9 else 0
    return (lp_val, ilp_val, ratio)


def calc_score(state, n):
    """
    decode => rectangles => solve coverage-based LP & ILP => ratio
    """
    bits_ = state.astype(int).tolist()
    rects = decode_rectangles_xywh_from_bits(bits_, n, bits_per_coord, coord_max)
    lp_val, ilp_val, ratio = solve_disjoint_rectangles(rects)
    return ratio, rects


def play_game(n_sess, actions, state_next, states, prob, step, total_score):
    """
    Standard approach:
      - sample an action for each session
      - if final step, compute reward
    """
    for i in range(n_sess):
        action = 1 if random.random() < prob[i] else 0
        actions[i][step - 1] = action

        old_state = states[i, :, step - 1]
        state_next[i] = old_state
        if action > 0:
            state_next[i][step - 1] = 1

        # clear time bit
        state_next[i][DECISIONS + step - 1] = 0
        terminal = (step == DECISIONS)
        if not terminal:
            state_next[i][DECISIONS + step] = 1
        else:
            # final => compute score
            ratio, _ = calc_score(state_next[i][:DECISIONS], num_rectangles)
            total_score[i] = ratio

        if not terminal:
            states[i, :, step] = state_next[i]
    return actions, state_next, states, total_score, terminal


def generate_session(agent, n_sess, verbose=1):
    obs_dim = 2 * DECISIONS
    states = np.zeros((n_sess, obs_dim, DECISIONS), dtype=int)
    actions = np.zeros((n_sess, DECISIONS), dtype=int)
    state_next = np.zeros((n_sess, obs_dim), dtype=int)
    total_score = np.zeros(n_sess)

    states[:, DECISIONS, 0] = 1
    step = 0
    pred_time = 0.0
    play_time = 0.0

    while True:
        step += 1
        t0 = time.time()
        probs = agent.predict(states[:, :, step - 1], batch_size=n_sess).flatten()
        pred_time += (time.time() - t0)

        t1 = time.time()
        actions, state_next, states, total_score, terminal = play_game(
            n_sess, actions, state_next, states, probs, step, total_score
        )
        play_time += (time.time() - t1)

        if terminal:
            break
    if verbose:
        print(f"Predict: {pred_time:.3f}, play: {play_time:.3f}")
    return states, actions, total_score


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    threshold = np.percentile(rewards_batch, percentile)
    length = len(states_batch)
    needed = length * (100.0 - percentile) / 100.0
    elite_states = []
    elite_actions = []
    for i in range(length):
        if rewards_batch[i] >= threshold - 1e-9:
            if (needed > 0) or (rewards_batch[i] > threshold + 1e-9):
                for st_i in range(DECISIONS):
                    elite_states.append(states_batch[i][:, st_i])
                    elite_actions.append(actions_batch[i][st_i])
                needed -= 1
    return np.array(elite_states, dtype=int), np.array(elite_actions, dtype=int)


def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
    threshold = np.percentile(rewards_batch, percentile)
    length = len(states_batch)
    needed = length * (100.0 - percentile) / 100.0
    super_states = []
    super_actions = []
    super_rewards = []
    for i in range(length):
        if rewards_batch[i] >= threshold - 1e-9:
            if (needed > 0) or (rewards_batch[i] > threshold + 1e-9):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
                needed -= 1
    return (
        np.array(super_states, dtype=int),
        np.array(super_actions, dtype=int),
        np.array(super_rewards, dtype=float),
    )


# ---------------------
# MAIN LOOP
# ---------------------
super_states = np.empty((0, observation_space, DECISIONS), dtype=int)
super_actions = np.empty((0, DECISIONS), dtype=int)
super_rewards = np.array([])

myRand = random.randint(0, 9999)

TARGET_RATIO = 1.4

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

best_state = super_actions[0]
_, best_rectangles = calc_score(best_state, num_rectangles)
plot_rectangles(best_rectangles)

