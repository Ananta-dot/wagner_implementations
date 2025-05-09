#######################################
# run_batcher_lis.py
#######################################

import code_template         # The "do not modify" RL template
import my_batcher_lis        # Our custom comparator & LIS module
from numba import njit
from keras.optimizers import SGD, Adam

# 1) Overwrite the placeholder calc_score in code_template
#    with the real one from my_batcher_lis
code_template.calc_score = my_batcher_lis.calc_score
code_template.jitted_calc_score = njit()(code_template.calc_score)

# 2) Make sure code_template sees our N and DECISIONS
code_template.N = my_batcher_lis.N
code_template.DECISIONS = my_batcher_lis.DECISIONS
# code_template defines:
#   observation_space = 2*DECISIONS
code_template.observation_space = 2 * code_template.DECISIONS

# 3) Rebuild and recompile the model with the new input shape 
#    (code_template.model is defined in code_template)
code_template.model.build((None, code_template.observation_space))
code_template.model.compile(
    loss="binary_crossentropy",
    optimizer=SGD(learning_rate=0.0001)
    # or you could use Adam, e.g. optimizer=Adam(learning_rate=0.0001)
)

# 4) Start the main training loop in code_template.
#    The file code_template.py might automatically do this at import,
#    or it might require a function call, e.g.:
# code_template.run_training()

if __name__ == "__main__":
    print("Running the Batcher-LIS RL training...")
    # If code_template has a main loop function, call it here:
    # code_template.run_training()
    # For now, if code_template automatically enters its loop on import,
    # there's nothing else we need to do.
    pass
