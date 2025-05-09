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

num_rectangles = 10

bits_per_coord = 2 * math.ceil(math.log2(num_rectangles))
DECISIONS = num_rectangles * 4 * bits_per_coord
observation_space = 2 * DECISIONS

LEARNING_RATE = 0.001
n_sessions = 400
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

def decode_rectangles_xywh_no_scaling(bit_array, num_rectangles, bits_per_coord):
    """
    Decodes 'bit_array' into a list of rectangles with no scaling or clamping:
      - x, y, w, h are directly taken as integers in [0..(1<<bits_per_coord)-1].
      - Then we define x2 = x + w, y2 = y + h, with no bounding checks/clamps.
    The final rectangle is ( (x, x2), (y, y2) ).

    If x2 < x or y2 < y (which can only happen if w<0 or h<0, but bits are unsigned),
    you'd get an empty or inverted rectangle. In principle we do not fix that here.
    """
    bits_per_rect = 4 * bits_per_coord
    needed_len = num_rectangles * bits_per_rect
    if len(bit_array) != needed_len:
        raise ValueError(f"Bit array length mismatch: want={needed_len}, got={len(bit_array)}")

    rectangles = []
    idx = 0
    for _ in range(num_rectangles):
        x_bits = bit_array[idx : idx + bits_per_coord]
        idx += bits_per_coord
        y_bits = bit_array[idx : idx + bits_per_coord]
        idx += bits_per_coord
        w_bits = bit_array[idx : idx + bits_per_coord]
        idx += bits_per_coord
        h_bits = bit_array[idx : idx + bits_per_coord]
        idx += bits_per_coord

        x_val = int("".join(str(b) for b in x_bits), 2)
        y_val = int("".join(str(b) for b in y_bits), 2)
        w_val = int("".join(str(b) for b in w_bits), 2)
        h_val = int("".join(str(b) for b in h_bits), 2)

        x_left = x_val
        x_right = x_val + w_val  # no clamp
        y_bottom = y_val
        y_top = y_val + h_val    # no clamp

        rectangles.append(((x_left, x_right), (y_bottom, y_top)))
    return rectangles

def plot_rectangles(rectangles):
    """
    Quick debug plot. If the bounding box is huge, you might not see them well.
    """
    fig, ax = plt.subplots(figsize=(8,8))
    xs, ys = [], []
    for i, ((x1, x2), (y1, y2)) in enumerate(rectangles):
        w = x2 - x1
        h = y2 - y1
        color = (random.random(), random.random(), random.random(), 0.4)
        rect = plt.Rectangle((x1,y1), w, h, edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(rect)
        cx, cy = (x1 + x2)/2, (y1 + y2)/2
        ax.text(cx, cy, f"{i+1}", fontsize=12, ha='center', va='center')
        xs.extend([x1,x2])
        ys.extend([y1,y2])
    if xs and ys:
        margin = 10
        ax.set_xlim(min(xs)-margin, max(xs)+margin)
        ax.set_ylim(min(ys)-margin, max(ys)+margin)
    ax.set_title("Rectangles (no scale/clamp)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.grid(True)
    plt.show()

def solve_disjoint_rectangles(rectangles):
    """
    Coverage-based approach:
      - Find bounding box from min_x..max_x, min_y..max_y among all rects
      - For each integer point, sum of chosen rects covering that point <= 1
      - Solve LP, then ILP
    """
    n_rect = len(rectangles)
    if n_rect == 0:
        return (0,0,0)

    min_x = min(r[0][0] for r in rectangles)
    max_x = max(r[0][1] for r in rectangles)
    min_y = min(r[1][0] for r in rectangles)
    max_y = max(r[1][1] for r in rectangles)

    # If rectangles can be huge or negative, this range might be big or inverted
    if max_x < min_x:
        # all rects are invalid or something
        return (0,0,0)
    if max_y < min_y:
        return (0,0,0)

    x_points = range(min_x, max_x+1)
    y_points = range(min_y, max_y+1)
    candidate_points = [(x,y) for x in x_points for y in y_points]

    constraints = []
    for (x, y) in candidate_points:
        covering = []
        for i, ((x1, x2),(y1,y2)) in enumerate(rectangles):
            if x1 < x < x2 and y1 < y < y2:
                covering.append(i)
        if len(covering)>1:
            constraints.append(covering)

    lp_val = 0.0
    ilp_val= 0.0

    # LP
    try:
        model_lp = gp.Model("RectLP")
        model_lp.Params.OutputFlag= 0
        x_lp = model_lp.addVars(n_rect, lb=0, ub=1, vtype=GRB.CONTINUOUS)
        model_lp.setObjective(gp.quicksum(x_lp[i] for i in range(n_rect)), GRB.MAXIMIZE)
        for covers in constraints:
            model_lp.addConstr(gp.quicksum(x_lp[i] for i in covers) <= 1)
        model_lp.optimize()
        if model_lp.status==GRB.OPTIMAL:
            lp_val = model_lp.objVal
    except Exception as e:
        print("LP Exception:", e)

    # ILP
    try:
        model_ilp = gp.Model("RectILP")
        model_ilp.Params.OutputFlag=0
        x_ilp = model_ilp.addVars(n_rect, vtype=GRB.BINARY)
        model_ilp.setObjective(gp.quicksum(x_ilp[i] for i in range(n_rect)), GRB.MAXIMIZE)
        for covers in constraints:
            model_ilp.addConstr(gp.quicksum(x_ilp[i] for i in covers) <=1)
        model_ilp.optimize()
        if model_ilp.status==GRB.OPTIMAL:
            ilp_val = model_ilp.objVal
    except Exception as e:
        print("ILP Exception:", e)

    ratio = lp_val / ilp_val if ilp_val>1e-9 else 0
    return (lp_val, ilp_val, ratio)

def calc_score(state, n):
    """
    Convert bitstring => rectangles => solve coverage-based disjointness => ratio
    """
    bits_array = state.astype(int).tolist()
    rects = decode_rectangles_xywh_no_scaling(bits_array, n, bits_per_coord)
    lp_val, ilp_val, ratio = solve_disjoint_rectangles(rects)
    return ratio, rects

def play_game(n_sess, actions, state_next, states, prob, step, total_score):
    """
    Standard RL bit approach:
      - each step sets 1 bit
      - at final step decode + compute ratio
    """
    for i in range(n_sess):
        action = 1 if random.random()<prob[i] else 0
        actions[i][step-1] = action
        old_state = states[i,:, step-1]
        state_next[i] = old_state
        if action>0:
            state_next[i][step-1] = 1

        # Clear time bit
        state_next[i][DECISIONS + step -1] = 0

        terminal = (step==DECISIONS)
        if not terminal:
            state_next[i][DECISIONS + step] = 1
        else:
            sc, _ = calc_score(state_next[i][:DECISIONS], num_rectangles)
            total_score[i] = sc

        if not terminal:
            states[i,:, step] = state_next[i]
    return actions, state_next, states, total_score, terminal


def generate_session(agent, n_sess, verbose=1):
    obs_dim = 2*DECISIONS
    states = np.zeros((n_sess, obs_dim, DECISIONS), dtype=int)
    actions= np.zeros((n_sess, DECISIONS), dtype=int)
    state_next= np.zeros((n_sess, obs_dim), dtype=int)
    total_score= np.zeros(n_sess)

    states[:, DECISIONS, 0] = 1
    step = 0
    pred_time=0.0
    play_time=0.0

    while True:
        step +=1
        t0 = time.time()
        probs = agent.predict(states[:,:,step-1], batch_size=n_sess).flatten()
        pred_time+= (time.time()-t0)

        t1 = time.time()
        actions, state_next, states, total_score, terminal = play_game(
            n_sess, actions, state_next, states, probs, step, total_score
        )
        play_time+=(time.time()-t1)
        if terminal:
            break
    if verbose:
        print(f"Predict= {pred_time:.3f}, play= {play_time:.3f}")
    return states, actions, total_score


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    threshold= np.percentile(rewards_batch, percentile)
    length= len(states_batch)
    needed = length*(100.0 - percentile)/100.0
    elite_states=[]
    elite_actions=[]
    for i in range(length):
        if rewards_batch[i]>= threshold-1e-9:
            if (needed>0) or (rewards_batch[i]> threshold+1e-9):
                for st_i in range(DECISIONS):
                    elite_states.append(states_batch[i][:, st_i])
                    elite_actions.append(actions_batch[i][st_i])
                needed-=1
    return np.array(elite_states,dtype=int), np.array(elite_actions,dtype=int)

def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
    threshold = np.percentile(rewards_batch, percentile)
    length= len(states_batch)
    needed= length*(100.0 - percentile)/100.0
    super_states=[]
    super_actions=[]
    super_rewards=[]
    for i in range(length):
        if rewards_batch[i]>=threshold-1e-9:
            if (needed>0) or (rewards_batch[i]>threshold+1e-9):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
                needed-=1
    return (np.array(super_states,dtype=int),
            np.array(super_actions,dtype=int),
            np.array(super_rewards,dtype=float))


# -------------
# MAIN RL LOOP
# -------------
super_states = np.empty((0, observation_space, DECISIONS), dtype=int)
super_actions= np.empty((0, DECISIONS), dtype=int)
super_rewards= np.array([])

MAX_ITER= 1000000
TARGET_RATIO= 1.4

for iteration in range(MAX_ITER):
    states_batch, actions_batch, rewards_batch = generate_session(model, n_sessions, verbose=0)

    # combine with super
    if len(super_states)>0:
        states_batch = np.concatenate([states_batch, super_states], axis=0)
        actions_batch= np.concatenate([actions_batch, super_actions], axis=0)
        rewards_batch= np.concatenate([rewards_batch, super_rewards])

    # select elites
    elite_states, elite_actions= select_elites(states_batch, actions_batch, rewards_batch, percentile=50)
    # select super
    s_states, s_actions, s_rewards= select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90)
    tri= [(s_states[i], s_actions[i], s_rewards[i]) for i in range(len(s_rewards))]
    tri.sort(key=lambda x:x[2], reverse=True)

    # train
    if len(elite_states)>0:
        model.fit(elite_states, elite_actions, verbose=0)

    super_states= np.array([x[0] for x in tri], dtype=int)
    super_actions= np.array([x[1] for x in tri], dtype=int)
    super_rewards= np.array([x[2] for x in tri], dtype=float)

    best_score= max(super_rewards) if len(super_rewards)>0 else 0
    mean_top50= np.mean(rewards_batch[-50:]) if len(rewards_batch)>=50 else np.mean(rewards_batch)
    print(f"Iter {iteration}, best={best_score:.3f}, mean_top50={mean_top50:.3f}")

    if best_score>= 1.3:
        print(f"Reached ratio >= {TARGET_RATIO} at iteration {iteration}. Stopping.")
        break

    if iteration %10 ==0:
        with open(f'best_species_{random.randint(0,9999)}.pkl','wb') as f:
            pickle.dump(super_actions, f)

print("\nTRAINING DONE.\n")
if len(super_actions)>0:
    best_bits= super_actions[0]
    ratio, best_rects = calc_score(best_bits, num_rectangles)
    print(f"BEST ratio= {ratio:.3f} (ILP coverage approach).")
    plot_rectangles(best_rects)
