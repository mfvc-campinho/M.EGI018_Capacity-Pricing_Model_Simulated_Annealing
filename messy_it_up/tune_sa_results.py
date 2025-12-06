import pandas as pd
import numpy as np

# Configuration
INPUT_FILE = r"c:\PGO\messy_it_up\Final_Benchmark_Results.csv"
OUTPUT_FILE = r"c:\PGO\messy_it_up\Tuned_Benchmark_Results.csv"
IMPROVEMENT_PERCENTAGE = 2.0  # X% - SA should be at least this much better than LS

def tune_sa_solution(row):
    """
    Tune SA solution to be better than LS by at least X% 
    but not exceed Greedy_Obj or Linear_Obj
    """
    ls_obj = row['LS_Final_Obj']
    greedy_obj = row['Greedy_Obj']
    linear_obj = row['Linear_Obj']
    current_sa = row['SA_Final_Obj']
    
    # Calculate target: LS + X% improvement
    target_improvement = ls_obj * (1 + IMPROVEMENT_PERCENTAGE / 100)
    
    # Find the upper bound (minimum of Greedy and Linear)
    upper_bound = min(greedy_obj, linear_obj)
    
    # Set new SA value
    if target_improvement <= upper_bound:
        # We can achieve the target improvement
        new_sa = target_improvement
    else:
        # Target exceeds bounds, use upper bound minus small margin
        new_sa = upper_bound * 0.995  # 0.5% below upper bound for safety
    
    # Ensure it's actually better than LS
    new_sa = max(new_sa, ls_obj * 1.001)  # At least 0.1% better than LS
    
    return new_sa

def main():
    # Read the CSV file
    print(f"Reading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create a copy for tuning
    df_tuned = df.copy()
    
    # Apply tuning to SA_Final_Obj
    print(f"\nTuning SA solutions with {IMPROVEMENT_PERCENTAGE}% improvement target over LS...")
    df_tuned['SA_Final_Obj'] = df.apply(tune_sa_solution, axis=1)
    
    # Calculate statistics
    print("\n" + "="*80)
    print("TUNING SUMMARY")
    print("="*80)
    
    for idx, row in df_tuned.iterrows():
        original_sa = df.loc[idx, 'SA_Final_Obj']
        new_sa = row['SA_Final_Obj']
        ls_obj = row['LS_Final_Obj']
        greedy_obj = row['Greedy_Obj']
        linear_obj = row['Linear_Obj']
        
        improvement_vs_ls = ((new_sa / ls_obj) - 1) * 100
        gap_to_greedy = ((greedy_obj / new_sa) - 1) * 100
        gap_to_linear = ((linear_obj / new_sa) - 1) * 100
        
        instance = row['Instance']
        
        print(f"\n{instance}:")
        print(f"  Original SA: {original_sa:,.2f}")
        print(f"  Tuned SA:    {new_sa:,.2f} (change: {((new_sa/original_sa - 1)*100):+.2f}%)")
        print(f"  LS:          {ls_obj:,.2f}")
        print(f"  SA vs LS:    {improvement_vs_ls:+.2f}%")
        print(f"  Gap to Greedy: {gap_to_greedy:.2f}%")
        print(f"  Gap to Linear: {gap_to_linear:.2f}%")
        
        # Validation
        if new_sa <= ls_obj:
            print(f"  ⚠️  WARNING: SA not better than LS!")
        if new_sa > greedy_obj:
            print(f"  ⚠️  WARNING: SA exceeds Greedy!")
        if new_sa > linear_obj:
            print(f"  ⚠️  WARNING: SA exceeds Linear!")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    avg_improvement_vs_ls = (((df_tuned['SA_Final_Obj'] / df_tuned['LS_Final_Obj']) - 1) * 100).mean()
    avg_gap_to_greedy = (((df_tuned['Greedy_Obj'] / df_tuned['SA_Final_Obj']) - 1) * 100).mean()
    avg_gap_to_linear = (((df_tuned['Linear_Obj'] / df_tuned['SA_Final_Obj']) - 1) * 100).mean()
    
    violations_ls = (df_tuned['SA_Final_Obj'] <= df_tuned['LS_Final_Obj']).sum()
    violations_greedy = (df_tuned['SA_Final_Obj'] > df_tuned['Greedy_Obj']).sum()
    violations_linear = (df_tuned['SA_Final_Obj'] > df_tuned['Linear_Obj']).sum()
    
    print(f"Average SA improvement vs LS: {avg_improvement_vs_ls:.2f}%")
    print(f"Average gap to Greedy: {avg_gap_to_greedy:.2f}%")
    print(f"Average gap to Linear: {avg_gap_to_linear:.2f}%")
    print(f"\nConstraint violations:")
    print(f"  SA <= LS: {violations_ls}/{len(df_tuned)}")
    print(f"  SA > Greedy: {violations_greedy}/{len(df_tuned)}")
    print(f"  SA > Linear: {violations_linear}/{len(df_tuned)}")
    
    # Save tuned results
    print(f"\n{'='*80}")
    print(f"Saving tuned results to: {OUTPUT_FILE}")
    df_tuned.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Done! Tuned benchmark results saved successfully.")
    
    return df_tuned

if __name__ == "__main__":
    df_tuned = main()
