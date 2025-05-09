import os
import re
import time
import shutil
import subprocess

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

N_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num_trials = 1

results_file = "results_10.csv"

# Write header including new "num_comparators" column.
with open(results_file, "w", encoding="utf-8") as f:
    f.write("N,trial,iteration_reached,runtime_seconds,num_comparators,bits,H,V,top_score\n")

def run_experiments():
    for N in N_values:
        # Compute the number of batcher comparators.
        # In try1.py, base_len is set to 4*n, so we mimic that here.
        base_len = 4 * N  
        comps = get_batcher_oe_comparators_py(base_len)
        num_comparators = len(comps)
        
        for trial in range(1, num_trials + 1):
            print(f"Running N={N}, trial={trial}...")
            shutil.copy("try1.py", "tmp_try1.py")
            with open("tmp_try1.py", "r", encoding="utf-8") as f:
                code = f.read()
            code = re.sub(
                r"^n = \d+",
                f"n = {N}",
                code,
                count=1,
                flags=re.MULTILINE
            )
            with open("tmp_try1.py", "w", encoding="utf-8") as f:
                f.write(code)

            start_time = time.time()
            process = subprocess.run(
                ["python3", "tmp_try1.py"],  
                capture_output=True,
                text=True
            )
            end_time = time.time()
            runtime_seconds = end_time - start_time

            stdout = process.stdout

            iter_match = re.search(r"Reached max score=.*?iteration=\s*(\d+)", stdout)
            if iter_match:
                iteration_reached = iter_match.group(1)
            else:
                iteration_reached = "(never reached best score)"

            sol_match = re.search(
                r"bits=\[(.*?)\],\s*score=([\d\.]+),\s*H=\[(.*?)\],\s*V=\[(.*?)\]",
                stdout,
                re.DOTALL
            )
            if sol_match:
                bits = sol_match.group(1).strip()
                top_score = sol_match.group(2).strip()
                H = sol_match.group(3).strip()
                V = sol_match.group(4).strip()
            else:
                bits = ""
                top_score = ""
                H = ""
                V = ""

            print(f"  -> Finished N={N}, trial={trial}, iteration={iteration_reached}, time={runtime_seconds:.2f}s, comparators={num_comparators}")

            with open(results_file, "a", encoding="utf-8") as f:
                f.write(f"{N},{trial},{iteration_reached},{runtime_seconds:.2f},{num_comparators},\"{bits}\",\"{H}\",\"{V}\",\"{top_score}\"\n")
            os.remove("tmp_try1.py")

    print(f"Done. See {results_file} for summarized results.")

run_experiments()
