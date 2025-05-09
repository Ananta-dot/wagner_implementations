import networkx as nx
import random
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model
from statistics import mean
import pickle
import time
import math
import matplotlib.pyplot as plt

def get_batcher_oe_comparators_py(n_):
    comps = []
    p = 1
    while p < n_:
        k = p
        while k>=1:
            j_start= k%p
            j_end= n_ -1 - k
            step_j= 2*k
            j= j_start
            while j<= j_end:
                i_max= min(k-1, n_ - j - k -1)
                for i in range(i_max+1):
                    left= (i+j)//(2*p)
                    right=(i+j+k)//(2*p)
                    if left==right:
                        comps.append((i+j, i+j+k))
                j+= step_j
            k//=2
        p*=2
    return comps

def build_base_array(n_):
    """
    For n_ = 3, returns [1,1,2,2,3,3]
    """
    arr = []
    for val in range(1, n_ + 1):
        arr.extend([val, val])
    return np.array(arr, dtype=int)

n = 2
base_len = 2 * n 
comps = get_batcher_oe_comparators_py(base_len)
m = len(comps)
DECISIONS = 2 * m
observation_space = 2 * DECISIONS

LEARNING_RATE= 0.0001
n_sessions= 1000
percentile= 93
super_percentile= 94
FIRST_LAYER_NEURONS=128
SECOND_LAYER_NEURONS=64
THIRD_LAYER_NEURONS=4

from keras import backend as K
K.clear_session()

model= Sequential()
model.add(Dense(FIRST_LAYER_NEURONS, activation="relu", input_shape=(observation_space,)))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate=LEARNING_RATE))

print(model.summary())

def intersect_interval_strict(ab, cd):
    """
    excludes boundary contact => if b==c or d==a => no overlap.
    """
    a, b = sorted(ab)
    c, d = sorted(cd)
    if b <= c or d <= a:
        return False
    return True

def intersect_strict(R1, R2):
    return (intersect_interval_strict(R1[0], R2[0]) and
            intersect_interval_strict(R1[1], R2[1]))

def apply_comps(baseA, bits, comps):
    arrc = baseA.copy()
    for i, bit in enumerate(bits):
        if bit==1:
            a, b = comps[i]
            arrc[a], arrc[b] = arrc[b], arrc[a]
    return arrc

def calc_score_and_arrays(state):
    """
    Build arrH and arrV using the modified base array (with 2*n elements),
    then build n rectangles (one for each unique number).
    
    Scoring:
      - For the circular chain: for each adjacent pair (including the pair connecting the last and the first)
        add +1 if they intersect.
      - For every non-adjacent pair, add +1 if they do NOT intersect.
    """
    # Split state into two halves for horizontal and vertical decisions.
    s1 = state[:m]
    s2 = state[m:]
    
    base_arr = build_base_array(n)  # Now produces an array of length 2*n, e.g. [1,1,2,2,3,3]
    arrH = apply_comps(base_arr, s1, comps)
    arrV = apply_comps(base_arr, s2, comps)
    
    # Build rectangles: one rectangle for each number 1..n.
    rects = []
    for i in range(1, n + 1):
        x_int = [index for index, item in enumerate(arrH) if item == i]
        y_int = [index for index, item in enumerate(arrV) if item == i]
        # Each rectangle is defined by its horizontal span and vertical span.
        rects.append([(min(x_int), max(x_int)), (min(y_int), max(y_int))])
    
    score = 0.0
    N = len(rects)  # This equals n.
    
    # --- 1. Check adjacent pairs in the circular chain ---
    if N == 2:
        # For 2 rectangles, there is only one unique pair.
        if intersect_strict(rects[0], rects[1]):
            score += 1
    else:
        # For n >= 3, check pairs (0,1), (1,2), ..., (N-2, N-1)
        for i in range(N - 1):
            if intersect_strict(rects[i], rects[i + 1]):
                score += 1
        # And the wrap-around pair (N-1, 0)
        if intersect_strict(rects[N - 1], rects[0]):
            score += 1
    
    # --- 2. Check non-adjacent pairs ---
    # For n = 2, there are no non-adjacent pairs.
    for i in range(N):
        for j in range(i + 1, N):
            # Skip adjacent pairs: for n>=3, these are (i, i+1) and (N-1, 0)
            if (N > 2) and ((j == i + 1) or (i == 0 and j == N - 1)):
                continue
            else:
                if not intersect_strict(rects[i], rects[j]):
                    score += 1
    
    return score, arrH, arrV

MAX_SCORE= int (n*((n-1)/2))

def play_game(n_sess, actions, state_next, states, prob, step, total_score,
              final_arrays_h, final_arrays_v):
    for i in range(n_sess):
        action = 1 if np.random.rand()< prob[i] else 0
        actions[i][step-1]= action

        for k in range(state_next.shape[1]):
            state_next[i,k]= states[i,k,step-1]
        if action>0:
            state_next[i, step-1]= 1
        state_next[i, DECISIONS + step-1]=0

        terminal= (step==DECISIONS)
        if not terminal:
            state_next[i, DECISIONS+ step]=1
        else:
            sc, arrH, arrV = calc_score_and_arrays(state_next[i][:DECISIONS])
            total_score[i]= sc
            final_arrays_h[i]= arrH
            final_arrays_v[i]= arrV

        if not terminal:
            for k in range(state_next.shape[1]):
                states[i,k,step]= state_next[i,k]
    return actions, state_next, states, total_score, terminal

def generate_session(agent, n_sess, verbose=1):
    states= np.zeros((n_sess, observation_space, DECISIONS), dtype=int)
    actions= np.zeros((n_sess, DECISIONS), dtype=int)
    state_next= np.zeros((n_sess, observation_space), dtype=int)
    total_score= np.zeros(n_sess, dtype=float)

    final_arrays_h= np.zeros((n_sess, 2*n), dtype=int)
    final_arrays_v= np.zeros((n_sess, 2*n), dtype=int)

    states[:,DECISIONS,0]=1
    step=0
    pred_time=0.0
    play_time=0.0

    while True:
        step+=1
        t0= time.time()
        prob= agent.predict(states[:,:,step-1], batch_size=n_sess)
        pred_time+= (time.time()-t0)

        t1= time.time()
        actions, state_next, states, total_score, terminal= play_game(
            n_sess, actions, state_next, states, prob.reshape(-1),
            step, total_score, final_arrays_h, final_arrays_v
        )
        play_time+= (time.time()-t1)

        if terminal:
            break

    if verbose:
        print(f"Predict: {pred_time:.3f}, play: {play_time:.3f}")

    return (states, actions, total_score, final_arrays_h, final_arrays_v)

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    needed= len(states_batch)*(100.0- percentile)/100.0
    rth= np.percentile(rewards_batch, percentile)
    es=[]
    ea=[]
    for i in range(len(states_batch)):
        if rewards_batch[i]>= (rth-1e-9):
            if (needed>0) or (rewards_batch[i]> rth+1e-9):
                for row in states_batch[i]:
                    es.append(row.tolist())
                for arow in actions_batch[i]:
                    ea.append(arow)
            needed-=1
    es= np.array(es,dtype=int)
    ea= np.array(ea,dtype=int)
    return es, ea

def select_super_sessions(states_batch, actions_batch, final_h_batch, final_v_batch, rewards_batch, percentile=90):
    needed= len(states_batch)*(100.0- percentile)/100.0
    rth= np.percentile(rewards_batch, percentile)
    s_states= []
    s_actions=[]
    s_fh=[]
    s_fv=[]
    s_rewards=[]
    for i in range(len(states_batch)):
        if rewards_batch[i]>= (rth-1e-9):
            if (needed>0) or (rewards_batch[i]> rth+1e-9):
                s_states.append(states_batch[i])
                s_actions.append(actions_batch[i])
                s_fh.append(final_h_batch[i])
                s_fv.append(final_v_batch[i])
                s_rewards.append(rewards_batch[i])
                needed-=1
    return (np.array(s_states,dtype=int),
            np.array(s_actions,dtype=int),
            np.array(s_fh,dtype=int),
            np.array(s_fv,dtype=int),
            np.array(s_rewards,dtype=float)
    )

super_states   = np.empty((0, DECISIONS, observation_space), dtype=int)
super_actions  = np.empty((0, DECISIONS), dtype=int)
super_final_h  = np.empty((0, 2*n), dtype=int)
super_final_v  = np.empty((0, 2*n), dtype=int)
super_rewards  = np.array([], dtype=float)

myRand= random.randint(0,1000)

sessgen_time=0.0
fit_time=0.0
score_time=0.0

for iteration in range(1000000):
    t0= time.time()
    sessions= generate_session(model, n_sessions, verbose=0)
    sessgen_time= time.time()-t0

    states_batch= np.array(sessions[0], dtype=int)
    actions_batch= np.array(sessions[1], dtype=int)
    rewards_batch= np.array(sessions[2], dtype=float)
    final_h_batch= np.array(sessions[3], dtype=int)
    final_v_batch= np.array(sessions[4], dtype=int)

    # reorder
    states_batch= np.transpose(states_batch, [0,2,1])

    # combine with super
    states_batch= np.append(states_batch, super_states, axis=0)
    actions_batch= np.append(actions_batch, super_actions, axis=0)
    final_h_batch= np.append(final_h_batch, super_final_h, axis=0)
    final_v_batch= np.append(final_v_batch, super_final_v, axis=0)
    rewards_batch= np.append(rewards_batch, super_rewards)

    randomcomp_time= time.time()-t0

    # pick elites
    t1= time.time()
    es,ea= select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile)
    sel_time1= time.time()-t1

    # pick super
    t2= time.time()
    s_states, s_actions, s_fh, s_fv, s_rewards= select_super_sessions(
        states_batch, actions_batch, final_h_batch, final_v_batch, rewards_batch,
        percentile=super_percentile
    )
    sel_time2= time.time()-t2

    # sort
    t3= time.time()
    tri= [(s_states[i], s_actions[i], s_fh[i], s_fv[i], s_rewards[i]) for i in range(len(s_rewards))]
    tri.sort(key=lambda x: x[4], reverse=True)
    sel_time3= time.time()-t3

    # train
    t4= time.time()
    if len(es)>0:
        model.fit(es, ea, verbose=0)
    fit_time= time.time()-t4

    t5= time.time()
    super_states  = np.array([x[0] for x in tri], dtype=int)
    super_actions = np.array([x[1] for x in tri], dtype=int)
    super_final_h = np.array([x[2] for x in tri], dtype=int)
    super_final_v = np.array([x[3] for x in tri], dtype=int)
    super_rewards = np.array([x[4] for x in tri], dtype=float)
    up_time= time.time()-t5

    # stats
    rewards_batch.sort()
    mean_all_reward= np.mean(rewards_batch[-100:]) if len(rewards_batch)>=100 else np.mean(rewards_batch)
    mean_best_reward= np.mean(super_rewards) if len(super_rewards)>0 else 0.0

    print(f"\n{iteration}. Best individuals: {np.flip(np.sort(super_rewards))}")
    print(f"Mean reward (top100): {mean_all_reward}")
    print("Timing =>",
          f"sessgen: {sessgen_time:.3f},",
          f"select1: {sel_time1:.3f},",
          f"select2: {sel_time2:.3f},",
          f"select3: {sel_time3:.3f},",
          f"fit: {fit_time:.3f},",
          f"upd: {up_time:.3f}")

    if len(super_rewards)>0 and max(super_rewards) >= MAX_SCORE:
        print(f"Reached max score= {MAX_SCORE} at iteration= {iteration}. Breaking.")
        break

    if iteration%20==1:
        with open(f'best_species_pickle_{myRand}.txt','wb') as fp:
            pickle.dump(super_actions, fp)
        with open(f'best_species_txt_{myRand}.txt','w') as f:
            for arr_ in super_actions:
                f.write(str(arr_.tolist())+"\n")
        with open(f'best_species_rewards_{myRand}.txt','w') as f:
            for rr in super_rewards:
                f.write(str(rr)+"\n")
        with open(f'best_100_rewards_{myRand}.txt','a') as f:
            f.write(str(mean_all_reward)+"\n")
        with open(f'best_elite_rewards_{myRand}.txt','a') as f:
            f.write(str(mean_best_reward)+"\n")

    if iteration%200==2 and len(super_actions)>0:
        with open(f'best_species_timeline_txt_{myRand}.txt','a') as f:
            f.write(str(super_actions[0].tolist())+"\n")

print("\n======================")
print("Training loop ended.")
print("Final top solutions in descending reward order:\n")
for i in range(len(super_rewards)):
    bits_str= super_actions[i].tolist()
    sc= super_rewards[i]
    arr_h= super_final_h[i].tolist()
    arr_v= super_final_v[i].tolist()
    print(f"bits={bits_str}, score={sc}, H={arr_h}, V={arr_v}")

avg= np.mean(super_rewards) if len(super_rewards)>0 else 0
print(f"\nAverage reward among final solutions= {avg}")
