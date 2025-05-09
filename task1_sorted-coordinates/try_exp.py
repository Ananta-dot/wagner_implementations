#!/usr/bin/env python3

import os
import re
import time
import shutil
import subprocess

# The values of N you want to test
N_values = [4, 5, 6, 7, 8, 9, 10]

# Number of trials for each N
num_trials = 1

results_file = "results.csv"

def run_experiments():
    results = []  # Will store tuples of (N, trial, iteration_reached, runtime_seconds)

    for N in N_values:
        for trial in range(1, num_trials + 1):
            # Print a message BEFORE running
            print(f"Running N={N}, trial={trial}...")

            # 1) Create a temporary copy of try2.py
            shutil.copy("try1.py", "tmp_try2.py")

            # 2) Replace `N = <number>` with `N = N`
            with open("tmp_try2.py", "r", encoding="utf-8") as f:
                code = f.read()

            # Replace the line that starts with `N = <digits>` once
            code = re.sub(
                r"^N = \d+",
                f"N = {N}",
                code,
                count=1,
                flags=re.MULTILINE
            )

            with open("tmp_try2.py", "w", encoding="utf-8") as f:
                f.write(code)

            # 3) Time the run
            start_time = time.time()

            process = subprocess.run(
                ["python3", "tmp_try2.py"],  # or just "python" depending on your environment
                capture_output=True,
                text=True
            )

            end_time = time.time()
            runtime_seconds = end_time - start_time

            # 4) Parse the iteration from the output (if any)
            stdout = process.stdout
            match = re.search(r"Reached max score=.*?iteration=\s*(\d+)", stdout)
            iteration_reached = match.group(1) if match else "(never reached best score)"

            # **Print a separate line AFTER each run finishes**
            print(f"  -> Finished N={N}, trial={trial}, iteration={iteration_reached}, time={runtime_seconds:.2f}s")

            # 5) Save results
            results.append((N, trial, iteration_reached, runtime_seconds))

            # 6) Remove temporary copy
            os.remove("tmp_try2.py")

    # Write all results as CSV
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("N,trial,iteration_reached,runtime_seconds\n")
        for (n_val, t_val, iters, rt) in results:
            f.write(f"{n_val},{t_val},{iters},{rt:.2f}\n")

    print(f"Done. See {results_file} for summarized results.")

if __name__ == "__main__":
    run_experiments()
