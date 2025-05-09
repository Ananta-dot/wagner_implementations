import random
import numpy as np
import pickle
import time
import math
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import multiprocessing
from functools import partial
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
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

# Problem size
n = 20
base_len = 2 * n
comps = get_batcher_oe_comparators_py(base_len)
m = len(comps)
DECISIONS = 2 * m
observation_space = 2 * DECISIONS

# Hyperparams
LEARNING_RATE = 0.0001
n_sessions = 4000
num_workers = 8  
percentile = 93
super_percentile = 94

K.clear_session()

def build_main_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(observation_space,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=LEARNING_RATE))
    return model

model = build_main_model()
print(model.summary())

def build_base_array(n_):
    arr = []
    for val in range(1, n_+1):
        arr.extend([val,val])
    return np.array(arr, dtype=int)

def apply_comps(baseA, bits, comps):
    arrc = baseA.copy()
    for i,bit in enumerate(bits):
        if bit==1:
            a,b = comps[i]
            arrc[a], arrc[b] = arrc[b], arrc[a]
    return arrc

def solve_disjoint_rectangles(rectangles):
    n_rect = len(rectangles)
    if n_rect==0:
        return (0.0, 0.0, 0.0)

    min_x = min(r[0][0] for r in rectangles)
    max_x = max(r[0][1] for r in rectangles)
    min_y = min(r[1][0] for r in rectangles)
    max_y = max(r[1][1] for r in rectangles)

    # integer points strictly inside
    x_points = list(range(min_x+1, max_x))
    y_points = list(range(min_y+1, max_y))
    constraints = []
    for x in x_points:
        for y in y_points:
            covering = []
            for i,rect in enumerate(rectangles):
                (x1,x2),(y1,y2)=rect
                if x1 < x < x2 and y1 < y < y2:
                    covering.append(i)
            if len(covering)>1:
                constraints.append(covering)

    # Gurobi multi-thread param
    # If you want to force single-thread or fewer threads:
    # e.g. threads=4 => model.setParam('Threads',4)
    # or let Gurobi decide.

    # LP
    lp_val=0.0
    try:
        model_lp = gp.Model('maxDisjointRectangles_LP')
        model_lp.setParam('OutputFlag',0)
        model_lp.setParam('Threads',4)  # optional
        x_lp = model_lp.addVars(n_rect, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='x')
        model_lp.setObjective(gp.quicksum(x_lp[i] for i in range(n_rect)), GRB.MAXIMIZE)
        for cset in constraints:
            model_lp.addConstr(gp.quicksum(x_lp[i] for i in cset) <= 1)
        model_lp.optimize()
        if model_lp.status==GRB.OPTIMAL:
            lp_val=model_lp.objVal
    except Exception as e:
        print('LP Exception:', e)

    # ILP
    ilp_val=0.0
    try:
        model_ilp= gp.Model('maxDisjointRectangles_ILP')
        model_ilp.setParam('OutputFlag',0)
        model_ilp.setParam('Threads',4)  # optional
        x_ilp= model_ilp.addVars(n_rect, vtype=GRB.BINARY)
        model_ilp.setObjective(gp.quicksum(x_ilp[i] for i in range(n_rect)), GRB.MAXIMIZE)
        for cset in constraints:
            model_ilp.addConstr(gp.quicksum(x_ilp[i] for i in cset)<=1)
        model_ilp.optimize()
        if model_ilp.status==GRB.OPTIMAL:
            ilp_val=model_ilp.objVal
    except Exception as e:
        print('ILP Exception:', e)

    ratio=0.0
    if ilp_val>1e-9:
        ratio=lp_val/ilp_val
    return (lp_val, ilp_val, ratio)

def calc_score(state, n):
    """
    Builds arrH, arrV from bits => rectangles => solve => ratio
    """
    s1= state[:m]
    s2= state[m:]
    base_arr= build_base_array(n)
    arrH= apply_comps(base_arr, s1, comps)
    arrV= apply_comps(base_arr, s2, comps)

    rectangles=[]
    for i in range(1, n+1):
        xidx= [k for k,val in enumerate(arrH) if val==i]
        yidx= [k for k,val in enumerate(arrV) if val==i]
        if len(xidx)<2 or len(yidx)<2:
            # degenerate
            rectangles.append(((0,0),(0,0)))
        else:
            x1,x2= min(xidx), max(xidx)
            y1,y2= min(yidx), max(yidx)
            rectangles.append(((x1,x2),(y1,y2)))

    lp_val, ilp_val, ratio= solve_disjoint_rectangles(rectangles)
    return ratio, arrH, arrV

def play_game_one_step(states, actions, state_next, total_score, step, prob):
    n_sess = len(prob)
    for i in range(n_sess):
        action = 1 if random.random()<prob[i] else 0
        actions[i][step-1]=action
        state_next[i] = states[i,:,step-1].copy()
        if action>0:
            state_next[i][step-1]=1

        state_next[i][DECISIONS+step-1]=0
        terminal=(step==DECISIONS)
        if not terminal:
            state_next[i][DECISIONS+step]=1
        else:
            # final => compute ratio
            sc,_,_= calc_score(state_next[i][:DECISIONS],n)
            total_score[i]=sc

        if not terminal:
            states[i,:,step]= state_next[i].copy()

    return actions, state_next, total_score, terminal

def run_session_chunk(model_weights, chunk_size):
    """
    Worker function:
    1. create local Keras model
    2. set weights
    3. run chunk_size sessions
    4. return states, actions, scores
    """
    # local model
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD

    # same architecture
    local_model = Sequential()
    local_model.add(Dense(128, activation='relu', input_shape=(observation_space,)))
    local_model.add(Dense(64, activation='relu'))
    local_model.add(Dense(4, activation='relu'))
    local_model.add(Dense(1, activation='sigmoid'))
    local_model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=LEARNING_RATE))

    # set weights
    local_model.set_weights(model_weights)

    obs_dim=2*DECISIONS
    states = np.zeros((chunk_size, obs_dim, DECISIONS), dtype=int)
    actions= np.zeros((chunk_size, DECISIONS), dtype=int)
    state_next= np.zeros((chunk_size, obs_dim), dtype=int)
    total_score= np.zeros(chunk_size, dtype=float)

    # init
    states[:, DECISIONS, 0]=1
    step=0

    while True:
        step+=1
        prob= local_model.predict(states[:,:,step-1], batch_size=chunk_size, verbose=0)
        actions, state_next, total_score, terminal= play_game_one_step(
            states, actions, state_next, total_score, step, prob.reshape(-1)
        )
        if terminal:
            break

    return states, actions, total_score

def generate_session_parallel(model, total_sessions, processes=4):
    """
    Distribute total_sessions among processes.
    """
    chunk_size= total_sessions//processes
    leftover= total_sessions % processes

    chunk_sizes = [chunk_size]*processes
    for i in range(leftover):
        chunk_sizes[i]+=1

    # snapshot weights
    weights= model.get_weights()

    pool= multiprocessing.Pool(processes)
    f= partial(run_session_chunk, weights)
    results= pool.map(f, chunk_sizes)
    pool.close()
    pool.join()

    states_list, actions_list, rewards_list = zip(*results)
    states_batch= np.concatenate(states_list, axis=0)
    actions_batch= np.concatenate(actions_list, axis=0)
    rewards_batch= np.concatenate(rewards_list, axis=0)
    return states_batch, actions_batch, rewards_batch

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    reward_threshold= np.percentile(rewards_batch, percentile)
    elite_states=[]
    elite_actions=[]
    needed= len(states_batch)*(100.0-percentile)/100.0
    for i in range(len(states_batch)):
        if rewards_batch[i]>= reward_threshold-1e-9:
            if needed>0 or (rewards_batch[i]>reward_threshold+1e-9):
                for t in range(DECISIONS):
                    elite_states.append(states_batch[i][:,t])
                    elite_actions.append(actions_batch[i][t])
            needed-=1
    elite_states=np.array(elite_states, dtype=int)
    elite_actions=np.array(elite_actions, dtype=int)
    return elite_states, elite_actions

def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
    reward_threshold= np.percentile(rewards_batch, percentile)
    super_states=[]
    super_actions=[]
    super_rewards=[]
    needed= len(states_batch)*(100.0-percentile)/100.0
    for i in range(len(states_batch)):
        if rewards_batch[i]>= reward_threshold-1e-9:
            if (needed>0) or (rewards_batch[i]> reward_threshold+1e-9):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
                needed-=1
    return (
        np.array(super_states,dtype=int),
        np.array(super_actions,dtype=int),
        np.array(super_rewards,dtype=float)
    )

def main():
    super_states= np.empty((0,2*DECISIONS,DECISIONS), dtype=int)
    super_actions= np.empty((0,DECISIONS), dtype=int)
    super_rewards= np.array([], dtype=float)

    myRand = random.randint(0,1000)
    TARGET_RATIO=1.4


    for iteration in range(1000000):
        t0= time.time()
        # Generate sessions in parallel
        states_batch, actions_batch, rewards_batch= generate_session_parallel(model, n_sessions, processes=num_workers)
        sessgen_time= time.time()-t0

        # Merge with super
        states_batch= np.concatenate([states_batch,super_states], axis=0)
        actions_batch= np.concatenate([actions_batch,super_actions], axis=0)
        rewards_batch= np.concatenate([rewards_batch,super_rewards])

        # Elites
        t1= time.time()
        elite_states, elite_actions= select_elites(states_batch, actions_batch, rewards_batch, percentile=50)
        sel_time1= time.time()-t1

        # Super
        t2= time.time()
        s_states, s_actions, s_rewards= select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90)
        sel_time2= time.time()-t2

        # Sort super
        t3= time.time()
        tri= sorted(zip(s_states, s_actions, s_rewards), key=lambda x:x[2], reverse=True)
        sel_time3= time.time()-t3

        # Fit
        t4= time.time()
        if len(elite_states)>0:
            model.fit(elite_states, elite_actions, verbose=0)
        fit_time= time.time()-t4

        # Update super
        t5= time.time()
        super_states= np.array([x[0] for x in tri], dtype=int)
        super_actions= np.array([x[1] for x in tri], dtype=int)
        super_rewards= np.array([x[2] for x in tri], dtype=float)
        up_time= time.time()-t5

        # Stats
        rewards_batch.sort()
        mean_all_reward= np.mean(rewards_batch[-100:]) if len(rewards_batch)>=100 else np.mean(rewards_batch)
        mean_best_reward= np.mean(super_rewards) if len(super_rewards)>0 else 0.0
        max_sr= max(super_rewards) if len(super_rewards)>0 else 0.0

        print(f"Iter {iteration}, best super ratio={max_sr:.3f}, top100 mean={mean_all_reward:.3f}")
        print("Timing =>", f"sessgen: {sessgen_time:.3f}", f"select1: {sel_time1:.3f}", f"select2: {sel_time2:.3f}", f"select3: {sel_time3:.3f}", f"fit: {fit_time:.3f}", f"upd: {up_time:.3f}")

        if len(super_rewards)>0 and max_sr>= TARGET_RATIO:
            print(f"Reached ratio >= {TARGET_RATIO} at iteration= {iteration}")
            break

        if iteration%20==1:
            with open(f"best_species_pickle_{myRand}.txt", 'wb') as fp:
                pickle.dump(super_actions, fp)

    print("\\n======================")
    print("Training loop ended.")
    print("Final top solutions in descending order:")
    for i in range(min(3, len(super_actions))):
        print(f"bits={super_actions[i].tolist()}, ratio={super_rewards[i]}")

if __name__ == "__main__":
    main()
