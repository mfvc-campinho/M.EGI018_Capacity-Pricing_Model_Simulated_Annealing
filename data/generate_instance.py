import pandas as pd
import numpy as np
import random
import os

def generate_strict_instance(filename, S_count, scale_factor=1.0):
    print(f"Generating Strict Instance (Inst40-like): {filename}")
    
    # --- 1. STRICT INST40 PARAMETERS ---
    G = 5
    T = 12  # Fixed at 12
    A = 3
    
    # Exact Costs from Inst40
    COS_g = [5.00,  4.50,  4.00,  3.50,  3.00]
    LEA_g = [1.50,  1.25,  1.00,  0.75,  0.50]
    OWN_g = [0.03, 0.025, 0.02, 0.015, 0.01]
    
    # Exact Prices from Inst40 (Excluding header row)
    PRICES = [
        [13.33, 13.32, 13.31, 13.30, 13.29],
        [21.66, 21.65, 21.64, 21.63, 21.62],
        [30.00, 29.99, 29.98, 29.97, 29.96]
    ]
    
    BUD = 500000 * scale_factor
    PYU = 1.0

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        
        # 1. UnitParameters
        params = {'G': G, 'S': S_count, 'A': A, 'T': T, 'PYU': PYU, 'BUD': BUD}
        pd.DataFrame.from_dict(params, orient='index', columns=[1])\
          .to_excel(writer, sheet_name='UnitParameters', header=False)

        # 2. RentalTypes (With Sparsity)
        rentals = []
        rental_id = 1
        max_duration = 3 
        
        # Target ~2400 rentals for S=4 (Similar size)
        # Full factorial for S=4 is ~6700. We need ~35% retention.
        # For larger S, we keep density similar or scale it.
        # Let's use a dynamic retention rate.
        total_possible = (S_count * S_count * G * T * max_duration)
        target_count = 2400 * scale_factor
        retention_rate = min(1.0, target_count / total_possible)

        print(f"  > Target Rentals: {target_count:.0f} (Retention: {retention_rate:.1%})")

        for s_start in range(1, S_count + 1):
            for s_end in range(1, S_count + 1):
                # 50% chance to skip self-loops (Structural Variability)
                if s_start == s_end and random.random() > 0.5: continue
                
                for group in range(1, G + 1):
                    for t_start in range(0, T):
                        for duration in range(1, max_duration + 1):
                            t_end = t_start + duration
                            if t_end <= T:
                                # SPARSITY FILTER
                                if random.random() < retention_rate:
                                    rentals.append([rental_id, s_start, s_end, t_start, t_end, group])
                                    rental_id += 1
                                
        df_rentals = pd.DataFrame(rentals, columns=['id', 'start_node', 'end_node', 'start_time', 'end_time', 'group'])
        df_rentals.to_excel(writer, sheet_name='RentalTypes', index=False)

        # 3. ParametersByGroup
        data_pbg = {
            'LEA_g': LEA_g,
            'LP_g': [8] * G,
            'COS_g': COS_g,
            'OWN_g': OWN_g
        }
        pd.DataFrame(data_pbg).T.to_excel(writer, sheet_name='ParametersByGroup', header=False)

        # 4. Prices
        pd.DataFrame(PRICES).to_excel(writer, sheet_name='Prices', header=False, index=False)
        P = len(PRICES)

        # 5. Demand (Inst40 Dist)
        demand_flat = []
        
        for r_row in rentals:
            # Log-Normal centered around 280
            base_dem = int(np.random.lognormal(mean=5.5, sigma=0.8))
            base_dem = max(0, min(1600, base_dem))
            
            for a in range(A + 1):
                for p in range(P):
                    factor = 1.0 - (p * 0.15)
                    val = int(base_dem * factor)
                    demand_flat.append(max(0, val))
                    
        pd.DataFrame(demand_flat, columns=["Demand"]).to_excel(writer, sheet_name='Demand', index=False)

        # 6. Upgrades
        upg = np.zeros((G, G), dtype=int)
        for i in range(G):
            for j in range(G):
                if j >= i: upg[i][j] = 1
        pd.DataFrame(upg).to_excel(writer, sheet_name='Upgrades', header=False, index=False)

        # 7. Transfer Costs & Times
        tc = np.zeros((S_count, S_count), dtype=float)
        tt = np.zeros((S_count, S_count), dtype=int)
        
        for i in range(S_count):
            for j in range(S_count):
                if i != j:
                    dist = abs(i - j) + 1
                    tc[i][j] = round(dist * 0.25, 2)
                    tt[i][j] = max(1, int(dist * 0.5))
                    
        pd.DataFrame(tc).to_excel(writer, sheet_name='TransferCosts', header=False, index=False)
        pd.DataFrame(tt).to_excel(writer, sheet_name='TransferTimes', header=False, index=False)
    
    print(f"  -> Saved {filename} ({len(rentals)} rentals).")

def run_batch_generation():
    if not os.path.exists("data"): os.makedirs("data")
    for i in range(1, 6):
        generate_strict_instance(f"data/Inst_Sim_{i}.xlsx", S_count=4, scale_factor=1.0)
    for i in range(1, 6):
        generate_strict_instance(f"data/Inst_Double_{i}.xlsx", S_count=6, scale_factor=2.0)
    for i in range(1, 3):
        generate_strict_instance(f"data/Inst_Quad_{i}.xlsx", S_count=8, scale_factor=4.0)

if __name__ == "__main__":
    run_batch_generation()