import pandas as pd
import time
import os
from new_linear import solve_instance

def run_batch_solver():
    # 1. Define the full list of files
    files = []
    files += [f"data/Inst_Sim_{i}.xlsx" for i in range(1, 6)]
    files += [f"data/Inst_Double_{i}.xlsx" for i in range(1, 6)]
    files += [f"data/Inst_Quad_{i}.xlsx" for i in range(1, 3)]
    
    output_file = "Final_Batch_Results.xlsx"
    results = []
    completed_files = set()

    # 2. Check for existing progress to RESUME
    if os.path.exists(output_file):
        try:
            print(f"Found existing results in {output_file}. Reading...")
            existing_df = pd.read_excel(output_file)
            
            # Load existing data so we don't lose it when we save later
            results = existing_df.to_dict('records')
            
            # Create a set of filenames that are already done
            if 'Instance' in existing_df.columns:
                completed_files = set(existing_df['Instance'].astype(str).tolist())
                print(f"-> identified {len(completed_files)} completed instances.")
        except Exception as e:
            print(f"Warning: Could not read existing file ({e}). Starting fresh.")

    print(f"\n--- Starting Batch Solve ---")
    print("Target Time Limit: 600 seconds (10 mins) per instance")

    # 3. Process Loop
    for filename in files:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Skipping {filename} (File Not Found)")
            continue
            
        # CHECK: Is this file already done?
        if filename in completed_files:
            print(f"Skipping {filename} (Already Completed)")
            continue

        # If not done, solve it
        start = time.time()
        
        # Call the solver
        result = solve_instance(filename)
        
        elapsed = time.time() - start
        result['Real_Time_s'] = round(elapsed, 1)
        
        # Add to results list
        results.append(result)
        
        # Save immediately (overwriting the file with the updated full list)
        pd.DataFrame(results).to_excel(output_file, index=False)

    print(f"\nBatch Solve Complete. All results saved to {output_file}")

if __name__ == "__main__":
    run_batch_solver()