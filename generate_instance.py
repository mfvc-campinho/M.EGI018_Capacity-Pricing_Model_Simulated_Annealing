# ==============================================================================
"""
[M.EGI018] Operations Management Project

INSTANCE GENERATOR - PROFITABLE ECONOMICS
-----------------------------------------
Fixes the "Zero Objective" issue by rebalancing costs vs revenue.
- Buying Cost: Reduced (20k -> 4k)
- Prices: Increased (100 -> 250)
- Result: It is now mathematically possible to make a profit.
"""
# ==============================================================================
import pandas as pd
import numpy as np
import random
import os

def generate_instance(filename, G, S, A, T, P, BUD, PYU):
    print(f"\nGenerating Profitable Instance: {filename}")
    print(f"  Config: {G} Groups, {S} Stations, {T} Periods")

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        
        # 1. UnitParameters
        params = {'G': G, 'S': S, 'A': A, 'T': T, 'PYU': PYU, 'BUD': BUD}
        pd.DataFrame.from_dict(params, orient='index', columns=[1])\
          .to_excel(writer, sheet_name='UnitParameters', header=False)

        # 2. RentalTypes
        rentals = []
        rental_id = 1
        max_duration = max(1, int(T * 0.25)) 
        
        for s_start in range(1, S + 1):
            for s_end in range(1, S + 1):
                if s_start == s_end and random.random() > 0.5: continue
                for group in range(1, G + 1):
                    for t_start in range(0, T):
                        for duration in range(1, max_duration + 1):
                            t_end = t_start + duration
                            if t_end <= T:
                                rentals.append([rental_id, s_start, s_end, t_start, t_end, group])
                                rental_id += 1
        df_rentals = pd.DataFrame(rentals, columns=['id', 'start_node', 'end_node', 'start_time', 'end_time', 'group'])
        df_rentals.to_excel(writer, sheet_name='RentalTypes', index=False)
        print(f"  -> {len(rentals)} Rental Types.")

        # 3. ParametersByGroup (ECONOMICS FIXED)
        # ---------------------------------------------------------
        data_pbg = {'LEA_g': [], 'LP_g': [], 'COS_g': [], 'OWN_g': []}
        
        # NEW COSTS:
        base_buy = 4000   # Was 20000
        base_lease = 100  # Was 400
        base_own = 10     # Was 50
        
        for g in range(G):
            mult = 1 + (g * 0.3)
            data_pbg['LEA_g'].append(int(base_lease * mult))
            data_pbg['LP_g'].append(int(T * 0.6)) 
            data_pbg['COS_g'].append(int(base_buy * mult))
            data_pbg['OWN_g'].append(int(base_own * mult))
        pd.DataFrame.from_dict(data_pbg, orient='index').to_excel(writer, sheet_name='ParametersByGroup', header=False)

        # 4. Prices (REVENUE INCREASED)
        # ---------------------------------------------------------
        prices = []
        base_p = 250 # Was 100
        
        for p in range(P):
            row = []
            for g in range(G):
                curr_p = base_p * (1 + (g * 0.3)) # Scale with group
                factor = 1.0 - (p * 0.2) 
                row.append(int(curr_p * factor))
            prices.append(row)
        pd.DataFrame(prices).to_excel(writer, sheet_name='Prices', header=False, index=False)

        # 5. Demand
        demand_flat = []
        for r_row in rentals:
            s_start = r_row[1]
            loc_bias = 1.5 if s_start == 1 else 1.0
            base_dem = int(30 * loc_bias)
            for a in range(A + 1):
                ant_factor = 1.0 + (a * 0.2)
                for p in range(P):
                    price_factor = 1 + (p * 0.5)
                    val = int(base_dem * ant_factor * price_factor * random.uniform(0.8, 1.2))
                    demand_flat.append(max(1, val))
        pd.DataFrame(demand_flat, columns=["Demand"]).to_excel(writer, sheet_name='Demand', index=False)

        # 6. Upgrades
        upg = np.zeros((G, G), dtype=int)
        for i in range(G):
            for j in range(G):
                if j >= i: upg[i][j] = 1
        pd.DataFrame(upg).to_excel(writer, sheet_name='Upgrades', header=False, index=False)

        # 7. Transfer Costs
        tc = np.zeros((S, S), dtype=int); tt = np.zeros((S, S), dtype=int)
        for i in range(S):
            for j in range(S):
                if i != j:
                    dist = abs(i - j) + 1
                    tc[i][j] = dist * 20
                    tt[i][j] = max(1, int(dist * 0.5))
        pd.DataFrame(tc).to_excel(writer, sheet_name='TransferCosts', header=False, index=False)
        pd.DataFrame(tt).to_excel(writer, sheet_name='TransferTimes', header=False, index=False)
    
    print(f"  -> Saved: {filename}")

# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists("data"): os.makedirs("data")

    # Inst41 (2x)
    generate_instance("data/Inst41.xlsx", G=5, S=4, A=3, T=24, P=3, BUD=5000000, PYU=5.0)
    
    # Inst42 (10x)
    generate_instance("data/Inst42.xlsx", G=5, S=10, A=3, T=20, P=3, BUD=20000000, PYU=5.0)