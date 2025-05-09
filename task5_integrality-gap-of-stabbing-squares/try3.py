import networkx as nx  # optional
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

import gurobipy as gp
from gurobipy import GRB

N = 25
square_size = 5
DECISIONS = int(N*(N-1)//2)

LEARNING_RATE   = 0.0001
n_sessions      = 1000
percentile      = 93
super_percentile= 94

FIRST_LAYER_NEURONS = 128
SECOND_LAYER_NEURONS= 64
THIRD_LAYER_NEURONS = 4

observation_space = 2*DECISIONS
len_game = DECISIONS

from keras import backend as K
K.clear_session()
model = Sequential()
model.add(Dense(FIRST_LAYER_NEURONS, activation="relu", input_shape=(observation_space,)))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate=LEARNING_RATE))
print(model.summary())


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
            if v>=xl and v<=xr:
                cExpr.addTerms(1.0, varV_ilp[v])
        for h in linesH:
            if h>=yb and h<=yt:
                cExpr.addTerms(1.0, varH_ilp[h])
        ilp.addConstr(cExpr >=1)
    ilp.setObjective(
        gp.quicksum(varV_ilp[v] for v in linesV) + gp.quicksum(varH_ilp[h] for h in linesH),
        GRB.MINIMIZE
    )
    ilp.optimize()
    if ilp.status==GRB.OPTIMAL:
        ILP_val= ilp.objVal
    else:
        ILP_val= None
    ilp.dispose()

    # 2) LP
    lp = gp.Model("ExactLP")
    lp.setParam('OutputFlag', 0)

    varV_lp= {}
    for v in linesV:
        varV_lp[v] = lp.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"V_{v}")
    varH_lp= {}
    for h in linesH:
        varH_lp[h] = lp.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"H_{h}")
    lp.update()

    for (xl, xr, yb, yt) in squares:
        cExpr= gp.LinExpr()
        for v in linesV:
            if v>=xl and v<=xr:
                cExpr.addTerms(1.0, varV_lp[v])
        for h in linesH:
            if h>=yb and h<=yt:
                cExpr.addTerms(1.0, varH_lp[h])
        lp.addConstr(cExpr >=1)

    lp.setObjective(
        gp.quicksum(varV_lp[v] for v in linesV) + gp.quicksum(varH_lp[h] for h in linesH),
        GRB.MINIMIZE
    )
    lp.optimize()
    if lp.status==GRB.OPTIMAL:
        LP_val= lp.objVal
    else:
        LP_val= None
    lp.dispose()

    return ILP_val, LP_val

def calc_score(state):
    for i in range(square_size):
        baseX = np.arange(1, N*square_size, square_size)
        baseY = np.arange(1, N*square_size, square_size)
    # must parse bits
    half= N
    x_bits= state[:half]
    y_bits= state[half:2*half]
    
    arrX= baseX.copy()
    for i, b in enumerate(x_bits):
        idx1= i%N
        idx2= (i+1)%N
        if b==1:
            tmp= arrX[idx1]
            arrX[idx1]= arrX[idx2]
            arrX[idx2]= tmp

    arrY= baseY.copy()
    for i, b in enumerate(y_bits):
        idx1= i%N
        idx2= (i+1)%N
        if b==1:
            tmp= arrY[idx1]
            arrY[idx2]= arrY[idx2]
            arrY[idx2]= tmp

    # squares
    squares=[]
    for i in range(N):
        xl= arrX[i]
        xr= xl+ square_size
        yl= arrY[i]
        yr= yl+ square_size
        squares.append((xl,xr,yl,yr))

    # gather lines from edges
    linesV_set= set()
    linesH_set= set()
    for (xl,xr,yl,yr) in squares:
        linesV_set.add(xl)
        linesV_set.add(xr)
        linesH_set.add(yl)
        linesH_set.add(yr)
    linesV = sorted(list(linesV_set))
    linesH = sorted(list(linesH_set))

    ILP_val, LP_val= solve_coverage_gurobi(squares, linesV, linesH)
    if ILP_val is None or LP_val is None or abs(LP_val)<1e-9:
        return 0.0
    return float(ILP_val)/ float(LP_val)

def play_game(n_sessions, actions, state_next, states, prob, step, total_score):
    for i in range(n_sessions):
        if random.random()< prob[i]:
            action=1
        else:
            action=0
        actions[i][step-1] = action

        # propagate old state
        for j in range(state_next.shape[1]):
            state_next[i,j] = states[i,j,step-1]

        # set bit
        if action>0:
            state_next[i,step-1]=1

        # turn off pointer
        state_next[i,DECISIONS + step-1]=0

        terminal= (step==DECISIONS)
        if not terminal:
            # set next pointer
            state_next[i,DECISIONS+ step]=1
        else:
            # final => compute coverage ratio
            total_score[i]= calc_score(state_next[i])
        # record sessions
        if not terminal:
            for j in range(state_next.shape[1]):
                states[i,j,step]= state_next[i,j]
    return actions, state_next, states, total_score, terminal

def generate_session(agent, n_sess, verbose=1):
    # shape => (n_sess, 2*DECISIONS, DECISIONS)
    states= np.zeros((n_sess, 2*DECISIONS, len_game), dtype=int)
    actions= np.zeros((n_sess, len_game), dtype=int)
    state_next= np.zeros((n_sess, 2*DECISIONS), dtype=int)
    total_score= np.zeros(n_sess)

    # pointer
    states[:, DECISIONS,0]=1

    step=0
    pred_time=0.0
    play_time=0.0
    while True:
        step+=1
        t0= time.time()
        prob= agent.predict(states[:,:,step-1], batch_size=n_sess).reshape(-1)
        pred_time+= time.time()-t0

        t1= time.time()
        actions, state_next, states, total_score, terminal = play_game(n_sess, actions, state_next, states, prob, step, total_score)
        play_time+= time.time()-t1

        if terminal:
            break

    if verbose:
        print(f"Predict: {pred_time:.3f}, play: {play_time:.3f}")
    return states, actions, total_score

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    from math import floor
    counter= n_sessions*(100.0- percentile)/100.0
    thresh= np.percentile(rewards_batch, percentile)
    elite_states=[]
    elite_actions=[]
    for i in range(len(states_batch)):
        if rewards_batch[i]>= thresh-1e-9:
            if counter>0 or (rewards_batch[i]> thresh+1e-9):
                for st in states_batch[i]:
                    elite_states.append(st)
                for ac in actions_batch[i]:
                    elite_actions.append(ac)
            counter-=1
    return np.array(elite_states,dtype=int), np.array(elite_actions,dtype=int)

def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
    counter= n_sessions*(100.0- percentile)/100.0
    thresh= np.percentile(rewards_batch, percentile)
    super_states=[]
    super_actions=[]
    super_rewards=[]
    for i in range(len(states_batch)):
        if rewards_batch[i]>= thresh-1e-9:
            if counter>0 or (rewards_batch[i]> thresh+1e-9):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
            counter-=1
    return (np.array(super_states,dtype=int), 
            np.array(super_actions,dtype=int),
            np.array(super_rewards,dtype=float))

super_states= np.empty((0,len_game,2*DECISIONS),dtype=int)
super_actions= np.empty((0,len_game), dtype=int)
super_rewards= np.array([])

myRand= random.randint(0,1000)

for iteration in range(1000000):
    # 1) generate sessions
    t0= time.time()
    sessions= generate_session(model,n_sessions,0)
    sessgen_time= time.time()-t0

    states_batch= sessions[0]  # (n_sess, 2*DECISIONS, DECISIONS)
    actions_batch= sessions[1] # (n_sess, DECISIONS)
    rewards_batch= sessions[2] # (n_sess,)

    states_batch= np.transpose(states_batch, (0,2,1))
    # shape => (n_sess, DECISIONS, 2*DECISIONS)

    states_batch= np.append(states_batch, super_states, axis=0)
    actions_batch= np.append(actions_batch, super_actions, axis=0)
    rewards_batch= np.append(rewards_batch, super_rewards)

    # 2) pick elites
    t1= time.time()
    elite_states, elite_actions= select_elites(states_batch, actions_batch, rewards_batch, percentile)
    sel_time1= time.time()- t1

    # 3) pick super sessions
    t2= time.time()
    s_states, s_actions, s_rewards= select_super_sessions(states_batch, actions_batch, rewards_batch, super_percentile)
    sel_time2= time.time()- t2

    # 4) sort super sessions
    t3= time.time()
    tri= [(s_states[i], s_actions[i], s_rewards[i]) for i in range(len(s_rewards))]
    tri.sort(key=lambda x: x[2], reverse=True)
    sel_time3= time.time()- t3

    # 5) fit from elites
    t4= time.time()
    if len(elite_states)>0:
        model.fit(elite_states, elite_actions, verbose=0)
    fit_time= time.time()- t4

    # 6) update super
    t5= time.time()
    super_states= np.array([x[0] for x in tri], dtype=int)
    super_actions= np.array([x[1] for x in tri], dtype=int)
    super_rewards= np.array([x[2] for x in tri], dtype=float)
    up_time= time.time()- t5

    # stats
    rewards_batch.sort()
    if len(rewards_batch)>=100:
        mean_all_reward= np.mean(rewards_batch[-100:])
    else:
        mean_all_reward= np.mean(rewards_batch)
    mean_best_reward= np.mean(super_rewards) if len(super_rewards)>0 else 0.0

    print(f"\nIter {iteration}. Best individuals: {np.flip(np.sort(super_rewards))}")
    print(f"Mean reward: {mean_all_reward:.3f}")
    print( "Timing =>"
          + f" sessgen: {sessgen_time:.3f},"
          + f" select1: {sel_time1:.3f},"
          + f" select2: {sel_time2:.3f},"
          + f" select3: {sel_time3:.3f},"
          + f" fit: {fit_time:.3f},"
          + f" upd: {up_time:.3f}"
    )

    # break condition
    if len(super_rewards)>0 and max(super_rewards)>1.5:
        print(f"Reached ratio >=1.4 at iteration= {iteration}. Breaking.")
        break

print("\n======================")
print("Training loop ended.")
print("Final top solutions in descending reward order:\n")
for i in range(min(3, len(super_actions))):
    bits_str= super_actions[i].tolist()
    sc= super_rewards[i]
    print(f"bits={bits_str}, ratio={sc}")
avg= np.mean(super_rewards) if len(super_rewards)>0 else 0
print(f"\nAverage reward among final solutions= {avg}")