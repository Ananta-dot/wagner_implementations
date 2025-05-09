import random
import numpy as np
import pickle
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from numba import njit

# Problem configuration
n = 4
base_length = 8 * n
target_score = (n ** 2) 

# Batcher network generation
def get_batcher_oe_comparators_py(n):
    comparators = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            j_start = k % p
            j_end = n - 1 - k
            step_j = 2 * k
            j = j_start
            while j <= j_end:
                i_max = min(k - 1, n - j - k - 1)
                for i in range(i_max + 1):
                    left = (i + j) // (2 * p)
                    right = (i + j + k) // (2 * p)
                    if left == right:
                        comparators.append((i + j, i + j + k))
                j += step_j
            k //= 2
        p *= 2
    return comparators

comparators_arr = np.array(get_batcher_oe_comparators_py(base_length), dtype=np.int64)
DECISIONS = len(comparators_arr)
observation_space = 2 * DECISIONS
len_game = DECISIONS

# Neural network
model = Sequential([
    Dense(128, activation="relu", input_shape=(observation_space,)),
    Dense(64, activation="relu"),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate=0.0001))

# Core functions
def create_base_array(n):
    blue_h = np.repeat(np.arange(1, n+1), 2)
    blue_v = np.repeat(np.arange(1, n+1), 2)
    red_h = np.repeat(np.arange(1, n+1), 2)
    red_v = np.repeat(np.arange(1, n+1), 2)
    return np.concatenate([blue_h, blue_v, red_h, red_v])

base_array = create_base_array(n)

@njit
def calc_score(state):
    arr_copy = base_array.copy()
    for i in range(DECISIONS):
        if state[i] == 1:
            a, b = comparators_arr[i][0], comparators_arr[i][1]
            arr_copy[a], arr_copy[b] = arr_copy[b], arr_copy[a]
    
    split_idx = len(arr_copy) // 2
    blue, red = arr_copy[:split_idx], arr_copy[split_idx:]
    
    coord_split = len(blue) // 2
    blue_h = blue[:coord_split].reshape(n, 2)
    blue_v = blue[coord_split:].reshape(n, 2)
    red_h = red[:coord_split].reshape(n, 2)
    red_v = red[coord_split:].reshape(n, 2)
    
    score = 0
    for i in range(n):
        bh0, bh1 = sorted(blue_h[i])
        bv0, bv1 = sorted(blue_v[i])
        for j in range(n):
            rh0, rh1 = sorted(red_h[j])
            rv0, rv1 = sorted(red_v[j])
            if (bh1 > rh0) and (rh1 > bh0) and (bv1 > rv0) and (rv1 > bv0):
                score += 1
                
    return score, arr_copy  # Return both score and final array

# Template functions
@njit
def play_game(n_sessions, actions, state_next, states, prob, step, total_score, final_arrays):
    for i in range(n_sessions):
        if np.random.rand() < prob[i]:
            action = 1
        else:
            action = 0
        actions[i][step-1] = action
        state_next[i] = states[i, :, step-1]

        if action > 0:
            state_next[i][step-1] = action
        state_next[i][DECISIONS + step-1] = 0
        if step < DECISIONS:
            state_next[i][DECISIONS + step] = 1
            
        if step == DECISIONS:
            score, arr = calc_score(state_next[i])
            total_score[i] = score
            final_arrays[i] = arr
        
        if not step == DECISIONS:
            states[i, :, step] = state_next[i]
    
    return actions, state_next, states, total_score, final_arrays, step == DECISIONS

def generate_session(agent, n_sess, verbose=1):
    states = np.zeros((n_sess, observation_space, DECISIONS), dtype=np.int64)
    actions = np.zeros((n_sess, DECISIONS), dtype=np.int64)
    state_next = np.zeros((n_sess, observation_space), dtype=np.int64)
    final_arrays = np.zeros((n_sess, base_length), dtype=np.int64)
    prob = np.zeros(n_sess)
    states[:, DECISIONS, 0] = 1
    step = 0
    total_score = np.zeros(n_sess)
    
    while True:
        step += 1
        prob = agent.predict(states[:, :, step-1], batch_size=n_sess)
        actions, state_next, states, total_score, final_arrays, terminal = play_game(
            n_sess, actions, state_next, states, prob, step, total_score, final_arrays
        )
        if terminal:
            break
            
    return states, actions, total_score, final_arrays

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    counter = len(states_batch) * (100.0 - percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = []
    elite_actions = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold-0.0000001:        
            if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
                for item in states_batch[i]:
                    elite_states.append(item.tolist())
                for item in actions_batch[i]:
                    elite_actions.append(item)			
            counter -= 1
    
    return np.array(elite_states, dtype=int), np.array(elite_actions, dtype=int)

def select_super_sessions(states_batch, actions_batch, rewards_batch, final_arrays_batch, percentile=90):
    counter = len(states_batch) * (100.0 - percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch, percentile)

    super_states = []
    super_actions = []
    super_rewards = []
    super_final_arrays = []
    
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold-0.0000001:
            if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
                super_final_arrays.append(final_arrays_batch[i])
                counter -= 1
    
    return (
        np.array(super_states, dtype=int),
        np.array(super_actions, dtype=int),
        np.array(super_rewards),
        np.array(super_final_arrays)
    )

# Training loop with final solution printing
def main_training_loop():
    n_sessions = 1000
    percentile = 93
    super_percentile = 94
    
    super_states = np.empty((0, len_game, observation_space), dtype=int)
    super_actions = np.empty((0, DECISIONS), dtype=int)
    super_rewards = np.array([])
    super_final_arrays = np.empty((0, base_length), dtype=int)
    
    myRand = random.randint(0, 1000)
    best_score = 0

    for iteration in range(1000000):
        # Session generation
        t_start = time.time()
        states_batch, actions_batch, rewards_batch, final_arrays_batch = generate_session(model, n_sessions)
        sess_time = time.time() - t_start

        # Data processing
        t_sel1 = time.time()
        states_batch = np.transpose(states_batch, axes=[0, 2, 1])
        states_batch = np.concatenate([states_batch, super_states], axis=0)
        actions_batch = np.concatenate([actions_batch, super_actions], axis=0)
        rewards_batch = np.concatenate([rewards_batch, super_rewards])
        final_arrays_batch = np.concatenate([final_arrays_batch, super_final_arrays], axis=0)
        randproc_time = time.time() - t_sel1

        # Selection phases
        t_sel1 = time.time()
        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)
        sel_time1 = time.time() - t_sel1

        t_sel2 = time.time()
        super_states, super_actions, super_rewards, super_final_arrays = select_super_sessions(
            states_batch, actions_batch, rewards_batch, final_arrays_batch, super_percentile
        )
        sel_time2 = time.time() - t_sel2

        t_sel3 = time.time()
        sorted_indices = np.argsort(-super_rewards)
        super_states = super_states[sorted_indices]
        super_actions = super_actions[sorted_indices]
        super_rewards = super_rewards[sorted_indices]
        super_final_arrays = super_final_arrays[sorted_indices]
        sel_time3 = time.time() - t_sel3

        # Model training
        t_fit = time.time()
        model.fit(elite_states, elite_actions, verbose=0)
        fit_time = time.time() - t_fit

        # Statistics
        t_upd = time.time()
        rewards_batch.sort()
        mean_all_reward = np.mean(rewards_batch[-100:])
        current_best = np.max(super_rewards)
        up_time = time.time() - t_upd

        # Iteration printout
        print(f"\nIter {iteration}. Best individuals: {np.flip(np.sort(super_rewards))}")
        print(f"Mean reward (top100): {mean_all_reward}")
        print("Timing =>",
              f"sessgen: {sess_time:.3f},",
              f"select1: {sel_time1:.3f},",
              f"select2: {sel_time2:.3f},",
              f"select3: {sel_time3:.3f},",
              f"fit: {fit_time:.3f},",
              f"upd: {up_time:.3f}")

        # Early stopping
        if current_best >= target_score:
            print(f"\nTarget score {target_score} achieved!")
            break

        # File output (keep existing logic here)

    # Final solutions printout
    print("\n======================")
    print("Training loop ended.")
    print("Final top solutions (arr, bits, score) in descending reward order:\n")
    for i in range(len(super_rewards)):
        arr_str = super_final_arrays[i].tolist()
        bits_str = super_actions[i].tolist()
        sc = super_rewards[i]
        print(f"arr={arr_str}, bits={bits_str}, score={sc}")

if __name__ == "__main__":
    main_training_loop()
