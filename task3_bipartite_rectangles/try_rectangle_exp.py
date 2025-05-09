import os
import re
import time
import shutil
import subprocess

N_values = [2, 3, 4, 5, 6, 7]
num_trials = 1
results_file = "results.csv"

with open(results_file, "w", encoding="utf-8") as f:
    f.write("N,trial,iteration_reached,runtime_seconds,bits,H,V,top_score\n")

def run_experiments():
    for N in N_values:
        for trial in range(1, num_trials + 1):
            print(f"Running N={N}, trial={trial}...")
            shutil.copy("run_file.py", "tmp_run_file.py")
            with open("tmp_run_file.py", "r", encoding="utf-8") as f:
                code = f.read()
            code = re.sub(
                r"^n = \d+",
                f"n = {N}",
                code,
                count=1,
                flags=re.MULTILINE
            )
            with open("tmp_try4_complete_1000000.py", "w", encoding="utf-8") as f:
                f.write(code)

            start_time = time.time()
            process = subprocess.run(
                ["python3", "tmp_try4_complete_1000000.py"], 
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

            print(f"  -> Finished N={N}, trial={trial}, iteration={iteration_reached}, time={runtime_seconds:.2f}s")
            with open(results_file, "a", encoding="utf-8") as f:
                f.write(f"{N},{trial},{iteration_reached},{runtime_seconds:.2f},\"{bits}\",\"{H}\",\"{V}\",\"{top_score}\"\n")
            os.remove("tmp_try4_complete_1000000.py")

    print(f"Done. See {results_file} for summarized results.")

run_experiments()
