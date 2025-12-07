import os
import time
import math
import random
import copy
import pandas as pd
from pyomo.environ import *

# Assuming these classes are in the same file or imported
# from run_all import InstanceData, PersistentEvaluator, generate_randomized_greedy, run_sa_algorithm

# NOTE: You must modify the `run_sa_algorithm` function in your original code 
# to accept `alpha` as a parameter.
# def run_sa_algorithm(data, evaluator_lp, initial_pricing, max_seconds=120, alpha=0.99): ...

def run_tuning_experiment():
    # 1. Define the parameters
    tuning_alphas = [0.95, 0.96, 0.97, 0.98, 0.99]
    tuning_time_limit = 300 # 5 minutes
    
    # 2. Select stratified instances (2 from each quartile)
    # Assuming files are named Inst1.xlsx, Inst2.xlsx...
    selected_indices = [5, 8, 15, 18, 25, 28, 35, 38]
    files = [f"data/Inst{i}.xlsx" for i in selected_indices]
    
    results = []

    print(f"--- Starting Parameter Tuning (Alpha) ---")
    print(f"Instances: {len(files)} | Alphas: {tuning_alphas} | Time: {tuning_time_limit}s")

    for filename in files:
        if not os.path.exists(filename):
            print(f"Skipping {filename} (Not Found)")
            continue
            
        print(f"\nProcessing Instance: {filename}")
        
        # Load Data once per instance
        data = InstanceData(filename)
        eval_lp = PersistentEvaluator(data, is_relaxed=True)
        
        # Generate a common start point to be fair
        random.seed(42)
        start_price = generate_randomized_greedy(data, randomness=0.30)
        start_obj, _, _ = eval_lp.solve_for_pricing(start_price)
        
        # Best Known (Load from your known results if calculating gap, 
        # otherwise just compare objective values)
        # For this script, we'll just save the Objective Value.
        
        for alpha in tuning_alphas:
            print(f"  > Testing Alpha: {alpha} ...", end="", flush=True)
            
            # Reset random seed for consistency across alphas if desired, 
            # or leave random to test robustness. 
            # Ideally, SA has internal randomness so we just run it.
            
            # RUN SA with specific Alpha
            # Note: You need to update run_sa_algorithm to accept 'alpha' argument
            sa_obj, sa_iters, sa_time = run_sa_algorithm(
                data, 
                eval_lp, 
                start_price, 
                max_seconds=tuning_time_limit,
                # alpha=alpha  <-- This needs to be passed to the function
            )
            
            print(f" Done. Obj: {sa_obj:,.0f}")
            
            results.append({
                'Instance': filename,
                'Alpha': alpha,
                'Objective': sa_obj,
                'Iterations': sa_iters
            })
            
        # Clean up
        eval_lp.close()
    
    # Save Results
    df = pd.DataFrame(results)
    df.to_excel("Tuning_Results.xlsx", index=False)
    print("\nTuning Complete. Saved to Tuning_Results.xlsx")

if __name__ == "__main__":
    run_tuning_experiment()