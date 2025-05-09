import random
import numpy as np
import pickle
import time
from statistics import mean
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from numba import njit

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

N = 4
_py_comps = get_batcher_oe_comparators_py(N)
comparators_arr = np.array(_py_comps, dtype=np.int64)
DECISIONS = comparators_arr.shape[0]
observation_space = 2 * DECISIONS  

LEARNING_RATE = 0.0001
n_sessions = 1000
percentile = 93
super_percentile = 94
FIRST_LAYER_NEURONS = 128
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

from keras import backend as K
K.clear_session()

model = Sequential()
model.add(Dense(FIRST_LAYER_NEURONS, activation="relu", input_shape=(observation_space,)))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate=LEARNING_RATE))
print(model.summary())

@njit
def calc_score(state, base_arr_h, base_arr_v):
    arr_h = base_arr_h.copy()
    arr_v = base_arr_v.copy()
    for i in range(DECISIONS):
        if state[i] == 1:
            a = comparators_arr[i, 0]
            b = comparators_arr[i, 1]
            # hor swap
            temp = arr_h[a]
            arr_h[a] = arr_h[b]
            arr_h[b] = temp
            # ver swap
            temp = arr_v[a]
            arr_v[a] = arr_v[b]
            arr_v[b] = temp

    score = 0
    for i in range(len(arr_h) - 1):
        if arr_h[i+1] > arr_h[i] and arr_v[i+1] > arr_v[i]:
            score += 1
    return arr_h, arr_v, score

@njit
def play_game_new(n_sess, actions, state_next, states, prob, step, total_score,
                  base_arr_h, base_arr_v, final_arr_h, final_arr_v):
    for i in range(n_sess):
        action = 1 if np.random.rand() < prob[i] else 0
        actions[i][step - 1] = action

        for k in range(state_next.shape[1]):
            state_next[i, k] = states[i, k, step - 1]

        if action == 1:
            state_next[i, step - 1] = 1

        state_next[i, DECISIONS + step - 1] = 0

        terminal = (step == DECISIONS)
        if not terminal:
            state_next[i, DECISIONS + step] = 1
        else:
            arr_h, arr_v, sc = calc_score(state_next[i], base_arr_h[i], base_arr_v[i])
            for x in range(N):
                final_arr_h[i, x] = arr_h[x]
                final_arr_v[i, x] = arr_v[x]
            total_score[i] = sc

        if not terminal:
            for k in range(state_next.shape[1]):
                states[i, k, step] = state_next[i, k]
    return actions, state_next, states, total_score, terminal

def generate_session_new(agent, n_sess, verbose=1):
    states = np.zeros((n_sess, observation_space, DECISIONS), dtype=np.int64)
    actions = np.zeros((n_sess, DECISIONS), dtype=np.int64)
    state_next = np.zeros((n_sess, observation_space), dtype=np.int64)
    total_score = np.zeros(n_sess, dtype=np.float64)

    base_arr_h = np.zeros((n_sess, N), dtype=np.int64)
    base_arr_v = np.zeros((n_sess, N), dtype=np.int64)
    for i in range(n_sess):
        for x in range(N):
            base_arr_h[i, x] = x + 1
            base_arr_v[i, x] = x + 1

    final_arr_h = np.zeros((n_sess, N), dtype=np.int64)
    final_arr_v = np.zeros((n_sess, N), dtype=np.int64)

    states[:, DECISIONS, 0] = 1

    step = 0
    pred_time = 0.0
    play_time = 0.0
    terminal = False

    while not terminal:
        step += 1
        t0 = time.time()
        prob = agent.predict(states[:, :, step - 1], batch_size=n_sess)
        pred_time += (time.time() - t0)

        t1 = time.time()
        actions, state_next, states, total_score, terminal = play_game_new(
            n_sess, actions, state_next, states, prob.reshape(-1), step,
            total_score, base_arr_h, base_arr_v, final_arr_h, final_arr_v
        )
        play_time += (time.time() - t1)

    if verbose:
        print(f"Predict: {pred_time:.3f}, play: {play_time:.3f}")

    return states, actions, total_score, final_arr_h, final_arr_v

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_states = []
    elite_actions = []
    needed = len(states_batch) * (100.0 - percentile) / 100.0
    for i in range(len(states_batch)):
        if rewards_batch[i] >= (reward_threshold - 1e-9):
            if (needed > 0) or (rewards_batch[i] > reward_threshold + 1e-9):
                for t in range(DECISIONS):
                    elite_states.append(states_batch[i, :, t])
                    elite_actions.append(actions_batch[i, t])
                needed -= 1
    return np.array(elite_states, dtype=np.int64), np.array(elite_actions, dtype=np.int64)

def select_super_sessions_new(states_batch, actions_batch, final_arr_h_batch, final_arr_v_batch, rewards_batch, percentile=90):
    reward_threshold = np.percentile(rewards_batch, percentile)
    super_states = []
    super_actions = []
    super_final_h = []
    super_final_v = []
    super_rewards = []
    needed = len(states_batch) * (100.0 - percentile) / 100.0
    for i in range(len(states_batch)):
        if rewards_batch[i] >= (reward_threshold - 1e-9):
            if (needed > 0) or (rewards_batch[i] > reward_threshold + 1e-9):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_final_h.append(final_arr_h_batch[i])
                super_final_v.append(final_arr_v_batch[i])
                super_rewards.append(rewards_batch[i])
                needed -= 1
    return (np.array(super_states, dtype=np.int64),
            np.array(super_actions, dtype=np.int64),
            np.array(super_final_h, dtype=np.int64),
            np.array(super_final_v, dtype=np.int64),
            np.array(super_rewards, dtype=np.float64))


def main_training_loop_new():
    super_states = np.empty((0, observation_space, DECISIONS), dtype=np.int64)
    super_actions = np.empty((0, DECISIONS), dtype=np.int64)
    super_final_h = np.empty((0, N), dtype=np.int64)
    super_final_v = np.empty((0, N), dtype=np.int64)
    super_rewards = np.array([], dtype=np.float64)
    myRand = random.randint(0, 1000)

    for iteration in range(1000):
        t0 = time.time()
        states_batch, actions_batch, rewards_batch, final_arr_h_batch, final_arr_v_batch = generate_session_new(model, n_sessions, verbose=0)
        sess_time = time.time() - t0

        states_batch = np.concatenate([states_batch, super_states], axis=0)
        actions_batch = np.concatenate([actions_batch, super_actions], axis=0)
        final_arr_h_batch = np.concatenate([final_arr_h_batch, super_final_h], axis=0)
        final_arr_v_batch = np.concatenate([final_arr_v_batch, super_final_v], axis=0)
        rewards_batch = np.concatenate([rewards_batch, super_rewards])

        t1 = time.time()
        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile)
        sel_time1 = time.time() - t1

        t2 = time.time()
        s_states, s_actions, s_final_h, s_final_v, s_rewards = select_super_sessions_new(
            states_batch, actions_batch, final_arr_h_batch, final_arr_v_batch, rewards_batch, percentile=super_percentile
        )
        sel_time2 = time.time() - t2

        t3 = time.time()
        tri = [(s_states[i], s_actions[i], s_final_h[i], s_final_v[i], s_rewards[i]) for i in range(len(s_rewards))]
        tri.sort(key=lambda x: x[4], reverse=True)
        sel_time3 = time.time() - t3

        t4 = time.time()
        model.fit(elite_states, elite_actions, verbose=0)
        fit_time = time.time() - t4

        t5 = time.time()
        super_states   = np.array([x[0] for x in tri], dtype=np.int64)
        super_actions  = np.array([x[1] for x in tri], dtype=np.int64)
        super_final_h  = np.array([x[2] for x in tri], dtype=np.int64)
        super_final_v  = np.array([x[3] for x in tri], dtype=np.int64)
        super_rewards  = np.array([x[4] for x in tri], dtype=np.float64)
        up_time = time.time() - t5

        rewards_batch.sort()
        mean_all_reward = np.mean(rewards_batch[-100:]) if len(rewards_batch) >= 100 else np.mean(rewards_batch)
        mean_best_reward = np.mean(super_rewards) if len(super_rewards) else 0

        print(f"\nIter {iteration}. Best individuals: {np.flip(np.sort(super_rewards))}")
        print(f"Mean reward (top100): {mean_all_reward}")
        print("Timing =>",
              f"sessgen: {sess_time:.3f},",
              f"select1: {sel_time1:.3f},",
              f"select2: {sel_time2:.3f},",
              f"select3: {sel_time3:.3f},",
              f"fit: {fit_time:.3f},",
              f"upd: {up_time:.3f}")
        print("Lowest best reward:", super_rewards[-1] if len(super_rewards) else None)

        if max(super_rewards) == (N - 1):
           print(f"Reached max score= {N-1} at iteration= {iteration}. Breaking.")
           print("Best reward:", super_rewards[-1])
           break

        if iteration % 20 == 1:
            with open(f'best_species_pickle_{myRand}.txt', 'wb') as fp:
                pickle.dump(super_actions, fp)
            with open(f'best_species_txt_{myRand}.txt', 'w') as f:
                for arr_ in super_actions:
                    f.write(str(arr_.tolist()) + "\n")
            with open(f'best_species_rewards_{myRand}.txt', 'w') as f:
                for rr in super_rewards:
                    f.write(str(rr) + "\n")
            with open(f'best_100_rewards_{myRand}.txt', 'a') as f:
                f.write(str(mean_all_reward) + "\n")
            with open(f'best_elite_rewards_{myRand}.txt', 'a') as f:
                f.write(str(mean_best_reward) + "\n")

        if iteration % 200 == 2 and len(super_actions) > 0:
            with open(f'best_species_timeline_txt_{myRand}.txt', 'a') as f:
                f.write(str(super_actions[0].tolist()) + "\n")

    print("\n======================")
    print("Training loop ended.")
    print("Final top solutions (horizontal, vertical, bits, score) in descending reward order:\n")
    for i in range(len(super_rewards)):
        h_arr = super_final_h[i].tolist()
        v_arr = super_final_v[i].tolist()
        bits_str = super_actions[i].tolist()
        sc = super_rewards[i]
        print(f"Horizontal: {h_arr}, Vertical: {v_arr}, bits: {bits_str}, score: {sc}")

    best_bits = super_actions[0]
    print(f"Average reward (score) over final solutions = {np.mean(super_rewards)}")

print(f"N = {N}, DECISIONS = {DECISIONS}")
print("Best possible score = ", N - 1)
main_training_loop_new()
