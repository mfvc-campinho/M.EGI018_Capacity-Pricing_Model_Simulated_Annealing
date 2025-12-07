import pandas as pd
import numpy as np

# Configuration
INPUT_FILE = r"c:\PGO\messy_it_up\Tuned_Benchmark_Results.csv"
OUTPUT_FILE = r"c:\PGO\messy_it_up\Improvement_Analysis.csv"

def compute_improvements(df):
    """
    Compute percentage improvements:
    - SA vs Constructive_Worse_Obj
    - LS vs Constructive_Worse_Obj
    - Linear vs Greedy_Obj
    """
    results = df[['Instance']].copy()
    
    # Calculate percentage improvements
    # Improvement = ((Better - Worse) / Worse) * 100
    results['SA_vs_Constructive_%'] = ((df['SA_Final_Obj'] - df['Constructive_Worse_Obj']) / df['Constructive_Worse_Obj']) * 100
    results['LS_vs_Constructive_%'] = ((df['LS_Final_Obj'] - df['Constructive_Worse_Obj']) / df['Constructive_Worse_Obj']) * 100
    results['Linear_vs_Greedy_%'] = ((df['Linear_Obj'] - df['Greedy_Obj']) / df['Greedy_Obj']) * 100
    
    # Add absolute values for reference
    results['Constructive_Worse_Obj'] = df['Constructive_Worse_Obj']
    results['SA_Final_Obj'] = df['SA_Final_Obj']
    results['LS_Final_Obj'] = df['LS_Final_Obj']
    results['Greedy_Obj'] = df['Greedy_Obj']
    results['Linear_Obj'] = df['Linear_Obj']
    
    return results

def display_table(df_results):
    """Display formatted table with improvement percentages"""
    print("\n" + "="*120)
    print("IMPROVEMENT ANALYSIS")
    print("="*120)
    print(f"\n{'Instance':<15} {'SA vs Constr':<15} {'LS vs Constr':<15} {'Linear vs Greedy':<18}")
    print(f"{'':<15} {'(%)':<15} {'(%)':<15} {'(%)':<18}")
    print("-"*120)
    
    for idx, row in df_results.iterrows():
        print(f"{row['Instance']:<15} {row['SA_vs_Constructive_%']:>14.2f} {row['LS_vs_Constructive_%']:>14.2f} {row['Linear_vs_Greedy_%']:>17.2f}")
    
    print("-"*120)
    print(f"{'AVERAGE':<15} {df_results['SA_vs_Constructive_%'].mean():>14.2f} {df_results['LS_vs_Constructive_%'].mean():>14.2f} {df_results['Linear_vs_Greedy_%'].mean():>17.2f}")
    print(f"{'MEDIAN':<15} {df_results['SA_vs_Constructive_%'].median():>14.2f} {df_results['LS_vs_Constructive_%'].median():>14.2f} {df_results['Linear_vs_Greedy_%'].median():>17.2f}")
    print(f"{'MIN':<15} {df_results['SA_vs_Constructive_%'].min():>14.2f} {df_results['LS_vs_Constructive_%'].min():>14.2f} {df_results['Linear_vs_Greedy_%'].min():>17.2f}")
    print(f"{'MAX':<15} {df_results['SA_vs_Constructive_%'].max():>14.2f} {df_results['LS_vs_Constructive_%'].max():>14.2f} {df_results['Linear_vs_Greedy_%'].max():>17.2f}")
    print("="*120)

def display_detailed_table(df_results):
    """Display detailed table with both percentages and absolute values"""
    print("\n" + "="*150)
    print("DETAILED IMPROVEMENT ANALYSIS")
    print("="*150)
    
    for idx, row in df_results.iterrows():
        print(f"\n{row['Instance']}:")
        print(f"  Constructive (Worse): {row['Constructive_Worse_Obj']:>15,.2f}")
        print(f"  SA Final:             {row['SA_Final_Obj']:>15,.2f}  (improvement: {row['SA_vs_Constructive_%']:+.2f}%)")
        print(f"  LS Final:             {row['LS_Final_Obj']:>15,.2f}  (improvement: {row['LS_vs_Constructive_%']:+.2f}%)")
        print(f"  Greedy:               {row['Greedy_Obj']:>15,.2f}")
        print(f"  Linear:               {row['Linear_Obj']:>15,.2f}  (improvement: {row['Linear_vs_Greedy_%']:+.2f}%)")
    
    print("\n" + "="*150)

def main():
    # Read the tuned benchmark results
    print(f"Reading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Compute improvements
    print("\nComputing improvements...")
    df_results = compute_improvements(df)
    
    # Display summary table
    display_table(df_results)
    
    # Display detailed table
    display_detailed_table(df_results)
    
    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 80)
    print(f"SA improvement over Constructive:")
    print(f"  Average: {df_results['SA_vs_Constructive_%'].mean():.2f}%")
    print(f"  Std Dev: {df_results['SA_vs_Constructive_%'].std():.2f}%")
    print(f"  Range:   [{df_results['SA_vs_Constructive_%'].min():.2f}%, {df_results['SA_vs_Constructive_%'].max():.2f}%]")
    
    print(f"\nLS improvement over Constructive:")
    print(f"  Average: {df_results['LS_vs_Constructive_%'].mean():.2f}%")
    print(f"  Std Dev: {df_results['LS_vs_Constructive_%'].std():.2f}%")
    print(f"  Range:   [{df_results['LS_vs_Constructive_%'].min():.2f}%, {df_results['LS_vs_Constructive_%'].max():.2f}%]")
    
    print(f"\nLinear improvement over Greedy:")
    print(f"  Average: {df_results['Linear_vs_Greedy_%'].mean():.2f}%")
    print(f"  Std Dev: {df_results['Linear_vs_Greedy_%'].std():.2f}%")
    print(f"  Range:   [{df_results['Linear_vs_Greedy_%'].min():.2f}%, {df_results['Linear_vs_Greedy_%'].max():.2f}%]")
    
    # Save results
    print(f"\n{'='*80}")
    print(f"Saving improvement analysis to: {OUTPUT_FILE}")
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Done! Improvement analysis saved successfully.")
    
    return df_results

if __name__ == "__main__":
    df_results = main()
