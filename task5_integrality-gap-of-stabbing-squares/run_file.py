import random
import time
import numpy as np
import pickle
from statistics import mean

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import math

import gurobipy as gp
from gurobipy import GRB, quicksum

NUM_SQUARES = 6

COORD_MAX = (NUM_SQUARES + 1) * NUM_SQUARES # 10 * NUM_SQUARES          
SQUARE_WIDTH = NUM_SQUARES # 0.2 * COORD_MAX
COORD_MAX = COORD_MAX - SQUARE_WIDTH  # so they fit in the range

BITS_PER_COORD = 3 * math.ceil(math.log2(NUM_SQUARES)) # # of bits for each x or y
print(BITS_PER_COORD)
BITS_PER_SQUARE = 2 * BITS_PER_COORD  # x + y
DECISIONS = NUM_SQUARES * BITS_PER_SQUARE
N = DECISIONS        

LEARNING_RATE    = 0.0001
n_sessions       = 1000
percentile       = 93
super_percentile = 94
FIRST_LAYER_NEURONS  = 128
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS  = 4
n_actions = 2  # 0/1 bits

observation_space = 2 * DECISIONS
len_game = DECISIONS

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(FIRST_LAYER_NEURONS,  activation="relu"))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.build((None, observation_space))
model.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate = LEARNING_RATE))
print(model.summary())

def decode_squares_from_bits(bit_array):
    """
    bit_array: 0-1 array of length DECISIONS (e.g. 240).
               We interpret each group of 12 bits => (x,y) for each square.
    Returns a list of squares: [(x1, x2, y1, y2), ...], all with side SQUARE_WIDTH.
    """
    squares = []
    bits_per_square = BITS_PER_SQUARE 
    n_squares = NUM_SQUARES           

    if len(bit_array) != DECISIONS:
        raise ValueError(f"Bit array length mismatch, expected {DECISIONS}, got {len(bit_array)}")

    for i in range(n_squares):
        start_idx = i * bits_per_square
        x_bits = bit_array[start_idx : start_idx + BITS_PER_COORD]
        y_bits = bit_array[start_idx + BITS_PER_COORD : start_idx + bits_per_square]

        x_val = int("".join(str(b) for b in x_bits), 2)
        y_val = int("".join(str(b) for b in y_bits), 2)

        max_int = (1 << BITS_PER_COORD) - 1  # e.g. 2^6 -1=63
        if max_int > 0:
            x_coord = int((x_val / max_int) * max(0, COORD_MAX - SQUARE_WIDTH))
            y_coord = int((y_val / max_int) * max(0, COORD_MAX - SQUARE_WIDTH))
        else:
            x_coord = 0
            y_coord = 0

        x1 = x_coord
        x2 = x_coord + SQUARE_WIDTH
        y1 = y_coord
        y2 = y_coord + SQUARE_WIDTH
        squares.append((x1, x2, y1, y2))
    return squares

def run_gurobi_stabbing(squares):
    """
    squares: list of (x1,x2,y1,y2)
    Returns (ILP_obj, LP_obj).
    """
    # gather candidate lines
    xcoords = set()
    ycoords = set()
    for (x1,x2,y1,y2) in squares:
        xcoords.add(x2)
        ycoords.add(y2)
    xcoords = sorted(list(xcoords))
    ycoords = sorted(list(ycoords))

    # ILP model
    model_ilp = gp.Model("stab_ILP")
    model_ilp.setParam("OutputFlag",0)
    # create binary vars
    V = model_ilp.addVars(len(xcoords), vtype=GRB.BINARY)
    H = model_ilp.addVars(len(ycoords), vtype=GRB.BINARY)
    # constraints
    for (x1,x2,y1,y2) in squares:
        x_idxs = [i for i,xv in enumerate(xcoords) if x1<=xv<=x2]
        y_idxs = [j for j,yv in enumerate(ycoords) if y1<=yv<=y2]
        model_ilp.addConstr(gp.quicksum(V[i] for i in x_idxs) + gp.quicksum(H[j] for j in y_idxs) >= 1)
    # objective
    model_ilp.setObjective(V.sum('*') + H.sum('*'), GRB.MINIMIZE)
    model_ilp.optimize()
    if model_ilp.status!= GRB.OPTIMAL:
        return (0,0)
    ilp_obj = model_ilp.objVal

    # LP model
    model_lp = model_ilp.relax()  # or create from scratch
    model_lp.setParam("OutputFlag",0)
    model_lp.optimize()
    if model_lp.status!= GRB.OPTIMAL:
        return (ilp_obj, 0)
    lp_obj = model_lp.objVal
    return (ilp_obj, lp_obj)

from numba import njit

def calc_score(state):
    """
    This is your 'reward' function in the template.
    Input: a 0-1 vector 'state' of length DECISIONS
    Output: the integrality gap (ILP/LP).
    """
    # decode squares from bits
    # state is a 0-1 array with length=DECISIONS
    bit_array = state.astype(int)  # ensure integer
    squares = decode_squares_from_bits(bit_array)

    # run gurobi
    ilp_val, lp_val = run_gurobi_stabbing(squares)
    if lp_val>1e-9:
        return ilp_val / lp_val
    else:
        return 0.0

jitted_calc_score = calc_score

def play_game(n_sessions, actions, state_next, states, prob, step, total_score):
    """
    For each session, sample an action for the current step.
    When step == DECISIONS, decode the candidate bit sequence and compute reward.
    """
    for i in range(n_sessions):
        action = 1 if random.random() < prob[i] else 0
        actions[i][step-1] = action
        state_next[i] = states[i,:,step-1]

        if (action > 0):
            state_next[i][step-1] = action
        state_next[i][DECISIONS + step-1] = 0
        terminal = (step == DECISIONS)
        if not terminal:
            state_next[i][DECISIONS + step] = 1

        if terminal:
            # slice the first DECISIONS bits only
            bit_array_240 = state_next[i][:DECISIONS]
            total_score[i] = jitted_calc_score(bit_array_240)
        if not terminal:
            states[i,:,step] = state_next[i]

    return actions, state_next, states, total_score, terminal

def generate_session(agent, n_sessions, verbose=1):
    """
    This is unchanged from your template except we do jitted_calc_score.
    """
    states = np.zeros([n_sessions, observation_space, len_game], dtype=int)
    actions= np.zeros([n_sessions, len_game], dtype=int)
    state_next= np.zeros([n_sessions, observation_space], dtype=int)
    prob= np.zeros(n_sessions)
    states[:, DECISIONS, 0] = 1
    step=0
    total_score = np.zeros(n_sessions)
    pred_time=0
    play_time=0

    while True:
        step+=1
        t0= time.time()
        prob = agent.predict(states[:,:,step-1], batch_size=n_sessions)
        pred_time+= time.time()-t0

        t0= time.time()
        actions, state_next, states, total_score, terminal = play_game(
            n_sessions, actions, state_next, states, prob, step, total_score
        )
        play_time+= time.time()-t0

        if terminal:
            break

    if verbose:
        print(f"Predict time: {pred_time:.3f}, play time: {play_time:.3f}")
    return states, actions, total_score

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    from numpy import percentile as np_percentile
    threshold= np_percentile(rewards_batch, percentile)
    length= len(states_batch)
    counter= length*(100-percentile)/100.0
    elite_states=[]
    elite_actions=[]
    for i in range(length):
        if rewards_batch[i]>= threshold-1e-9:
            if (counter>0) or (rewards_batch[i]>= threshold+1e-9):
                for st in states_batch[i]:
                    elite_states.append(st.tolist())
                for ac in actions_batch[i]:
                    elite_actions.append(ac)
                counter-=1
    elite_states= np.array(elite_states,dtype=int)
    elite_actions= np.array(elite_actions,dtype=int)
    return elite_states, elite_actions

def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
    from numpy import percentile as np_percentile
    threshold= np_percentile(rewards_batch, percentile)
    length= len(states_batch)
    counter= length*(100-percentile)/100.0
    super_states=[]
    super_actions=[]
    super_rewards=[]
    for i in range(length):
        if rewards_batch[i]>= threshold-1e-9:
            if (counter>0) or (rewards_batch[i]>= threshold+1e-9):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
                counter-=1
    super_states= np.array(super_states,dtype=int)
    super_actions= np.array(super_actions,dtype=int)
    super_rewards= np.array(super_rewards,dtype=float)
    return super_states, super_actions, super_rewards

super_states  = np.empty((0,len_game,observation_space), dtype=int)
super_actions = np.array([], dtype=int)
super_rewards = np.array([])
import time, pickle

myRand = random.randint(0,1000)

for iteration in range(1000000):
    t0= time.time()
    sessions = generate_session(model,n_sessions, verbose=0)
    sess_time= time.time()-t0

    states_batch= np.array(sessions[0],dtype=int)
    actions_batch= np.array(sessions[1],dtype=int)
    rewards_batch= np.array(sessions[2],dtype=float)
    states_batch= np.transpose(states_batch,(0,2,1))

    if len(super_states)>0:
        states_batch= np.append(states_batch, np.array(super_states), axis=0)
        actions_batch= np.append(actions_batch, np.array(super_actions), axis=0)
        rewards_batch= np.append(rewards_batch, np.array(super_rewards))

    elite_states, elite_actions= select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile)
    s_sess= select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile)
    super_sess_list= [(s_sess[0][i], s_sess[1][i], s_sess[2][i]) for i in range(len(s_sess[2]))]
    super_sess_list.sort(key=lambda x:x[2], reverse=True)

    model.fit(elite_states, elite_actions, verbose=0)

    super_states  = [x[0] for x in super_sess_list]
    super_actions = [x[1] for x in super_sess_list]
    super_rewards = [x[2] for x in super_sess_list]

    rewards_batch.sort()
    mean_all= np.mean(rewards_batch[-100:]) if len(rewards_batch)>=100 else np.mean(rewards_batch)
    mean_best= np.mean(super_rewards) if len(super_rewards)>0 else 0
    best_sorted= np.flip(np.sort(super_rewards))

    print(f"\nIteration {iteration}: Best rewards: {best_sorted}")
    print(f"Mean reward: {mean_all:.3f}, Session gen time: {sess_time:.3f}")

    # early stop if we exceed 2.0, for example
    if len(best_sorted)>0 and best_sorted[0]>=1.5:
        target = 1.5
        print(f"Reached max score= {target} at iteration= {iteration}. Breaking.")
        break

    if iteration %20 ==1:
        with open(f'best_species_pickle_{myRand}.txt','wb') as fp:
            pickle.dump(super_actions, fp)
        with open(f'best_species_txt_{myRand}.txt','w') as f:
            for item in super_actions:
                f.write(str(item)); f.write("\n")
        with open(f'best_species_rewards_{myRand}.txt','w') as f:
            for item in super_rewards:
                f.write(str(item)); f.write("\n")
        with open(f'best_100_rewards_{myRand}.txt','a') as f:
            f.write(str(mean_all)+"\n")
        with open(f'best_elite_rewards_{myRand}.txt','a') as f:
            f.write(str(mean_best)+"\n")

    if iteration %200 ==2:
        with open(f'best_species_timeline_txt_{myRand}.txt','a') as f:
            if len(super_actions)>0:
                f.write(str(super_actions[0])+"\n")


print("\n======================")
print("Training loop ended.")
print("Final top solutions in descending reward order:\n")
for i in range(min(3,len(super_actions))):
    bits_str= super_actions[i].tolist()
    sc= super_rewards[i]
    print(f"bits={bits_str}, ratio={sc}")
avg= np.mean(super_rewards) if len(super_rewards)>0 else 0
print(f"\nAverage reward among final solutions= {avg}")
