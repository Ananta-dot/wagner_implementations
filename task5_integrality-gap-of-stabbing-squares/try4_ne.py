import random
import time
import numpy as np
import pickle
import math
from statistics import mean
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from numba import njit
from gurobipy import Model, GRB, quicksum

nRect = 10

DECISIONS = 4 * nRect   # total length of the "word" = 2nRect for horizontal + 2nRect for vertical

# RL hyperparameters
LEARNING_RATE = 0.0001
n_sessions = 1000       # number of new sessions per iteration
percentile = 93         # top 100-X percentile we learn from
super_percentile = 94   # top 100-X percentile that survives to next iteration

FIRST_LAYER_NEURONS = 128
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

n_actions = 2  # We only generate bits {0,1}

# The RL template expects an "observation space" of size 2*DECISIONS:
#  - First DECISIONS bits store the partially built sequence.
#  - Next DECISIONS bits store a one-hot position pointer.
observation_space = 2 * DECISIONS
len_game = DECISIONS


model = Sequential()
model.add(Dense(FIRST_LAYER_NEURONS,  activation="relu"))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS,  activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.build((None, observation_space))
model.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate=LEARNING_RATE))

print(model.summary())

def build_rectangles_from_dag_sequence(seq, nRect):
    """
    Interpret seq (length 2*nRect) as a topological sort of the DAG for either
    horizontal or vertical intervals. The DAG enforces:
      - Must pick l_i before r_i for each i.
      - Must pick l_j in order before l_i, etc. (the chain structure).
    
    This function returns a list of intervals. For horizontal usage, these
    intervals are (left_i, right_i). For vertical usage, (top_i, bottom_i).

    We'll do a simple approach:
      - We label the nodes: l1, l2, ..., ln, r1, r2, ..., rn in "DAG order."
      - '0' means "pick the next L if available"
      - '1' means "pick the next R if available"

    Then we pair up each (l_i, r_i) by the order in which they appear in seq.
    """
    # We'll keep track of the order in which l_i and r_i are chosen.
    # Because each i must appear exactly once as an 'l_i' and once as an 'r_i',
    # the final arrangement determines the intervals.

    # Example labeling for nRect=4:
    # DAG horizontal: l1->l2->l3->l4 and edges from each li->ri, etc.
    # We'll read the bits in seq, each '0' means pick the smallest unpicked li,
    # each '1' means pick the smallest unpicked ri that can be picked.
    
    l_used = [False]*nRect
    r_used = [False]*nRect
    
    # We'll store the final "appearance order" for each l_i and r_i.
    # lPos[i] = index in the sequence (0..2nRect-1) where l_i occurred
    # rPos[i] = index in the sequence (0..2nRect-1) where r_i occurred
    lPos = [-1]*nRect
    rPos = [-1]*nRect
    
    next_l = 0  # which l_i is next in line if we see a '0'
    # 'next_l' goes from 0..nRect-1
    # We'll only pick an 'r_i' if its corresponding l_i was already used.

    for index, bit in enumerate(seq):
        if bit == 0:
            # pick the next l_i if possible
            if next_l < nRect:
                l_used[next_l] = True
                lPos[next_l] = index
                next_l += 1
            else:
                # if we run out of l_i's, any leftover bits do nothing special,
                # but let's keep it simple and ignore it or skip.
                pass
        else:
            # pick the earliest i s.t. l_i is used but r_i not used
            for i in range(nRect):
                if l_used[i] and not r_used[i]:
                    r_used[i] = True
                    rPos[i] = index
                    break
                # skip until we find a valid i

    # Now we have each i's lPos[i], rPos[i]. We'll interpret them as coordinates:
    # We'll define the coordinate of i-th interval as (lPos[i], rPos[i]) in sorted order
    # to ensure left < right if used for horizontal, top < bottom if used for vertical.
    intervals = []
    for i in range(nRect):
        left_coord  = min(lPos[i], rPos[i])
        right_coord = max(lPos[i], rPos[i])
        intervals.append( (float(left_coord), float(right_coord)) )
    return intervals


def combine_intervals_as_squares(horizontal_intervals, vertical_intervals):
    """
    We have nRect horizontal intervals (left, right) and nRect vertical intervals (top, bottom).
    We'll pair them up index-wise to form squares:
        rect[i] = [ (left_i, top_i), (right_i, bottom_i) ] in the plane.
    Because we want them all to be "the same size," we interpret the difference
    (right_i - left_i) = (bottom_i - top_i). Here, we just trust the user that 
    the DAG approach ensures consistent sizes. If not exact, adapt logic as needed.

    Return a list of rectangles in the form:
      [ (L_i, R_i, T_i, B_i), ... ]
    or equivalently any structure you prefer.
    """
    # We'll assume they are "the same size" because the DAG constraints enforce it
    # in a real scenario. For demonstration, we just pair them by index.
    # Each i corresponds to the i-th square.
    rectangles = []
    for i in range(nRect):
        L = horizontal_intervals[i][0]
        R = horizontal_intervals[i][1]
        T = vertical_intervals[i][0]
        B = vertical_intervals[i][1]
        # Ensure T < B, L < R
        if T > B:
            T, B = B, T
        if L > R:
            L, R = R, L
        rectangles.append( (L, R, T, B) )
    return rectangles


###############################################################################
# 2. Gurobi ILP/LP: Solve the rectangle-stabbing problem integrally and fractionally
###############################################################################

def run_gurobi(rectangles):
    """
    Solve the rectangle stabbing problem:
      - We want a minimal set of horizontal/vertical lines that intersects each rectangle.
      - We'll do it as:
           minimize sum(H_j) + sum(V_k)
        subject to
           for each rectangle i, sum(H_j that stabs i) + sum(V_k that stabs i) >= 1
           0 <= H_j, V_k <= 1
      - We'll do one solve with integer constraints (ILP) => ILP_value
      - We'll do one solve with continuous constraints (LP) => LP_value

    rectangles is a list of (L, R, T, B).

    We'll let "candidate lines" be exactly the unique x-coordinates and y-coordinates
    from the set of rectangle edges. That is enough for an optimal solution in 
    axis-aligned stabbing.

    Return: (ILP_value, LP_value).
    """

    # 1) Extract unique x-coordinates and y-coordinates
    x_coords = set()
    y_coords = set()
    for (L, R, T, B) in rectangles:
        x_coords.add(L); x_coords.add(R)
        y_coords.add(T); y_coords.add(B)
    x_coords = sorted(list(x_coords))
    y_coords = sorted(list(y_coords))

    # index them
    x_index = { x_coords[i] : i for i in range(len(x_coords)) }
    y_index = { y_coords[j] : j for j in range(len(y_coords)) }

    # We'll define a function that, for rectangle i, returns which x-index or y-index lines
    # can stab it. A line "x = x_k" stabs rectangle i if L <= x_k <= R.
    # Similarly for "y = y_k".

    def x_candidates_for_rect(R):
        indices = []
        for x in range(len(x_coords)):
            if x_coords[x]== R:
                indices.append(x)
        return indices

    def y_candidates_for_rect(T):
        indices = []
        for y in range(len(y_coords)):
            if y_coords[y] == T:
                indices.append(y)
        return indices

    # ------------------------- ILP Solve -------------------------
    ILP_model = Model("stab_ILP")
    ILP_model.setParam("OutputFlag", 0)  # turn off Gurobi printing

    # Add variables: H_j, V_k are binary
    H = ILP_model.addVars(len(y_coords), vtype=GRB.BINARY, name="H")
    V = ILP_model.addVars(len(x_coords), vtype=GRB.BINARY, name="V")

    # Constraints: for each rectangle i, sum(H_j that stabs i) + sum(V_k that stabs i) >= 1
    for (L, R, T, B) in rectangles:
        x_cands = x_candidates_for_rect(R)
        y_cands = y_candidates_for_rect(T)
        ILP_model.addConstr(
            quicksum(V[xi] for xi in x_cands) + quicksum(H[yi] for yi in y_cands) >= 1
        )

    # Objective: minimize sum(H_j) + sum(V_k)
    ILP_model.setObjective(quicksum(H[j] for j in range(len(y_coords))) +
                           quicksum(V[k] for k in range(len(x_coords))),
                           GRB.MINIMIZE)
    ILP_model.optimize()
    ILP_value = ILP_model.objVal

    # ------------------------- LP Solve -------------------------
    # We'll copy the same model but relax variables to continuous in [0,1].
    LP_model = Model("stab_LP")
    LP_model.setParam("OutputFlag", 0)

    # Now H, V are continuous in [0,1]
    H_lp = LP_model.addVars(len(y_coords), vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="H_lp")
    V_lp = LP_model.addVars(len(x_coords), vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="V_lp")

    for (L, R, T, B) in rectangles:
        x_cands = x_candidates_for_rect(R)
        y_cands = y_candidates_for_rect(T)
        LP_model.addConstr(
            quicksum(V_lp[xi] for xi in x_cands) + quicksum(H_lp[yi] for yi in y_cands) >= 1
        )

    LP_model.setObjective(quicksum(H_lp[j] for j in range(len(y_coords))) +
                          quicksum(V_lp[k] for k in range(len(x_coords))),
                          GRB.MINIMIZE)
    LP_model.optimize()
    LP_value = LP_model.objVal

    return (ILP_value, LP_value)

@njit
def calc_score(state):
    """
    Numba-compiled wrapper to call a slower function that does Gurobi solves.
    HOWEVER, we cannot call Gurobi directly from @njit code. Instead, we'll do:
      - parse the bit sequence in Python,
      - call a helper (Python) function for Gurobi,
      - combine results,
      - return the ratio.
    
    But the RL code uses this function inside a numba routine (play_game).
    So we trick it: we store the data we need in a global variable or
    use a separate pipeline. Typically, you can't directly do Gurobi calls
    from within Numba. The usual approach is: if we only need the final
    reward, we compute it outside. 
    
    For demonstration, let's do a "mock" approach. We'll do a global reference
    or pass the final state out. This is a known limitation. 
    In a real system, you might re-architect to call the Gurobi step after the
    episodes finish. 
    """
    # For now, just return a dummy float. The real logic is in 'calc_score_nojit'.
    return 0.0

###############################################################################
# Because of the Gurobi-Numba mismatch, we do an additional function
# that does the real work in pure Python. The RL template calls `calc_score`
# at the end of each session, but we will do a final pass that re-evaluates 
# the constructed states in pure Python, or we forcibly do the Gurobi solve 
# right before returning from play_game. 
#
# For demonstration here, we'll adjust the RL loop so that after a session is 
# done, we call a Python function that replicates the logic of `calc_score`.
###############################################################################

def calc_score_python(state):
    """
    Full Python version of the reward function.
    - state: array of length 2*DECISIONS, but the first DECISIONS bits
      store the final object.

    Steps:
      1. Extract first half => seqAll (length=DECISIONS).
      2. Split into horizontal bits, vertical bits.
      3. Build intervals => squares => run gurobi => get ILP/LP => ratio
    """
    seqAll = state[:DECISIONS]
    # For convenience:
    #  first 2*nRect bits => horizontal seq
    #  next 2*nRect bits => vertical seq
    #  total = 4*nRect = DECISIONS

    half = 2*nRect
    seq_horizontal = seqAll[:half]
    seq_vertical   = seqAll[half:]

    # Convert them to intervals
    horiz_intervals = build_rectangles_from_dag_sequence(seq_horizontal, nRect)
    vert_intervals  = build_rectangles_from_dag_sequence(seq_vertical,   nRect)

    # Pair them up => squares
    rectangles = combine_intervals_as_squares(horiz_intervals, vert_intervals)

    # Solve ILP, LP
    ilp_val, lp_val = run_gurobi(rectangles)
    if lp_val < 1e-9:
        # avoid division by zero if LP is 0
        return float(ilp_val)  # or large
    integrality_gap = ilp_val / lp_val

    return float(integrality_gap)


###############################################################################
# 4. RL Template Code (Unchanged except for hooking in calc_score_python)
###############################################################################

def play_game(n_sessions, actions, state_next, states, prob, step, total_score):
    """
    The main loop for generating one step in each session.
    We'll modify it so that if 'terminal', we compute the real Gurobi-based
    reward in Python (calc_score_python) for each final state.
    """
    for i in range(n_sessions):
        if np.random.rand() < prob[i]:
            action = 1
        else:
            action = 0
        actions[i][step-1] = action
        state_next[i] = states[i,:,step-1]

        if (action > 0):
            state_next[i][step-1] = action
        state_next[i][DECISIONS + step-1] = 0
        if (step < DECISIONS):
            state_next[i][DECISIONS + step] = 1
        # check if terminal
        terminal = (step == DECISIONS)
        if terminal:
            # Instead of jitted_calc_score, do real Python-based:
            total_score[i] = calc_score_python(state_next[i])

        if not terminal:
            states[i,:,step] = state_next[i]
    return actions, state_next, states, total_score, terminal

# We can't compile the above with Numba due to Gurobi calls in the final step.


def generate_session(agent, n_sessions, verbose=1):
    """
    Play n_session games using agent neural network.
    Terminate when games finish.
    """
    states = np.zeros([n_sessions, observation_space, len_game], dtype=int)
    actions = np.zeros([n_sessions, len_game], dtype=int)
    state_next = np.zeros([n_sessions, observation_space], dtype=int)
    prob = np.zeros(n_sessions)
    states[:, DECISIONS, 0] = 1
    step = 0
    total_score = np.zeros([n_sessions])
    pred_time = 0
    play_time = 0

    while True:
        step += 1
        tic = time.time()
        prob = agent.predict(states[:,:,step-1], batch_size=n_sessions)
        pred_time += time.time() - tic

        tic = time.time()
        actions, state_next, states, total_score, terminal = play_game(
            n_sessions, actions, state_next, states, prob, step, total_score
        )
        play_time += time.time() - tic

        if terminal:
            break

    if verbose:
        print("Predict: " + str(pred_time) + ", play: " + str(play_time))
    return states, actions, total_score


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    This is standard cross-entropy method logic.
    """
    counter = n_sessions * (100.0 - percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = []
    elite_actions = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold - 1e-7:
            if (counter > 0) or (rewards_batch[i] >= reward_threshold + 1e-7):
                for item in states_batch[i]:
                    elite_states.append(item.tolist())
                for item in actions_batch[i]:
                    elite_actions.append(item)
                counter -= 1
    elite_states = np.array(elite_states, dtype=int)
    elite_actions = np.array(elite_actions, dtype=int)
    return elite_states, elite_actions


def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
    """
    Pick the sessions that survive to the next generation.
    """
    counter = n_sessions * (100.0 - percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch, percentile)

    super_states = []
    super_actions = []
    super_rewards = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold - 1e-7:
            if (counter > 0) or (rewards_batch[i] >= reward_threshold + 1e-7):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
                counter -= 1
    super_states = np.array(super_states, dtype=int)
    super_actions = np.array(super_actions, dtype=int)
    super_rewards = np.array(super_rewards)
    return super_states, super_actions, super_rewards

super_states = np.empty((0, len_game, observation_space), dtype=int)
super_actions = np.array([], dtype=int)
super_rewards = np.array([])
sessgen_time = 0
fit_time = 0
score_time = 0

myRand = random.randint(0, 1000)  # used in the filenames

for i in range(1000000):  # up to 1,000,000 generations
    # 1) Generate new sessions
    tic = time.time()
    sessions = generate_session(model, n_sessions, verbose=0)
    sessgen_time = time.time() - tic
    tic = time.time()

    states_batch = np.array(sessions[0], dtype=int)  # shape (n_sessions, observation_space, len_game)
    actions_batch = np.array(sessions[1], dtype=int) # shape (n_sessions, len_game)
    rewards_batch = np.array(sessions[2])           # shape (n_sessions,)

    # reorder states_batch to shape (n_sessions, len_game, observation_space)
    states_batch = np.transpose(states_batch, axes=[0, 2, 1])

    # append best from previous generation
    states_batch = np.append(states_batch, super_states, axis=0)
    if i > 0:
        actions_batch = np.append(actions_batch, np.array(super_actions), axis=0)
    rewards_batch = np.append(rewards_batch, super_rewards)

    randomcomp_time = time.time() - tic
    tic = time.time()

    # 2) Select elites to train on
    elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile)
    select1_time = time.time() - tic

    # 3) Select top sessions to carry over
    tic = time.time()
    super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile)
    select2_time = time.time() - tic

    tic = time.time()
    super_sessions = [(super_sessions[0][j], super_sessions[1][j], super_sessions[2][j]) 
                      for j in range(len(super_sessions[2]))]
    super_sessions.sort(key=lambda x: x[2], reverse=True)
    select3_time = time.time() - tic

    # 4) Train on elites
    tic = time.time()
    model.fit(elite_states, elite_actions, verbose=0)
    fit_time = time.time() - tic

    tic = time.time()

    # 5) Update super_states, super_actions, super_rewards
    super_states  = [super_sessions[j][0] for j in range(len(super_sessions))]
    super_actions = [super_sessions[j][1] for j in range(len(super_sessions))]
    super_rewards = [super_sessions[j][2] for j in range(len(super_sessions))]

    rewards_batch.sort()
    mean_all_reward = np.mean(rewards_batch[-100:])   # average of the top 100 from *all*
    mean_best_reward = np.mean(super_rewards)         # average of survivors

    score_time = time.time() - tic

    # 6) Print progress
    print("\n" + str(i) +  ". Best individuals: " + str(np.flip(np.sort(super_rewards))))
    print("Mean reward: " + str(mean_all_reward) +
          "\nSessgen: " + str(sessgen_time) + 
          ", other: " + str(randomcomp_time) + 
          ", select1: " + str(select1_time) + 
          ", select2: " + str(select2_time) + 
          ", select3: " + str(select3_time) +  
          ", fit: " + str(fit_time) + 
          ", score: " + str(score_time))

    # 7) Save results periodically
    if (i % 20 == 1):
        with open('best_species_pickle_'+str(myRand)+'.txt', 'wb') as fp:
            pickle.dump(super_actions, fp)
        with open('best_species_txt_'+str(myRand)+'.txt', 'w') as f:
            for item in super_actions:
                f.write(str(item))
                f.write("\n")
        with open('best_species_rewards_'+str(myRand)+'.txt', 'w') as f:
            for item in super_rewards:
                f.write(str(item))
                f.write("\n")
        with open('best_100_rewards_'+str(myRand)+'.txt', 'a') as f:
            f.write(str(mean_all_reward)+"\n")
        with open('best_elite_rewards_'+str(myRand)+'.txt', 'a') as f:
            f.write(str(mean_best_reward)+"\n")

    if (i % 200 == 2):
        with open('best_species_timeline_txt_'+str(myRand)+'.txt', 'a') as f:
            f.write(str(super_actions[0]))
            f.write("\n")
