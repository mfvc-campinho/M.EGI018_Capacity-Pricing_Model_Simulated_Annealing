# ==============================================================================
"""
[M.EGI018] Operations Management Project

CONSTRUCTIVE HEURISTIC - BUDGET AWARE
-------------------------------------
1. Calculates 'Ideal' Fleet to meet all demand (Flow Simulation).
2. Calculates Total Cost of that fleet.
3. If Cost > Budget, proportionally scales down the fleet to fit.
   -> Guarantees a FEASIBLE starting solution.
"""
# ==============================================================================
import pandas as pd
import numpy as np
import os
import math

def solve_budget_heuristic(filename, output_folder="constructive_solutions"):
    inst_name = os.path.basename(filename).replace(".xlsx", "")
    print(f"\nProcessing {inst_name} (Budget-Aware)...")

    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_path = os.path.join(output_folder, f"Solution_{inst_name}.xlsx")

    try:
        xls = pd.ExcelFile(filename)

        # --- HELPER ---
        def get_clean_matrix(sheet_name):
            df = pd.read_excel(xls, sheet_name, header=None)
            return df.apply(pd.to_numeric, errors='coerce').dropna(how='all').dropna(how='all', axis=1).values

        # 1. Load Params (Get Budget!)
        up_df = pd.read_excel(xls, 'UnitParameters', header=None, index_col=0)
        G = int(up_df.loc['G'].iloc[0]); S = int(up_df.loc['S'].iloc[0])
        A = int(up_df.loc['A'].iloc[0]); T = int(up_df.loc['T'].iloc[0])
        BUD = float(up_df.loc['BUD'].iloc[0]) # <--- CRITICAL

        # 2. Load Costs
        pbg_df = pd.read_excel(xls, 'ParametersByGroup', header=None, index_col=0)
        def get_gp(name, dtype=float):
            vals = pbg_df.loc[name].values.flatten()
            return {g: dtype(vals[g]) if g < len(vals) else dtype(vals[0]) for g in range(G)}
        COS_g = get_gp('COS_g') # Buying Cost

        # 3. Load Rentals
        rentals_df = pd.read_excel(xls, 'RentalTypes')
        r_pot = rentals_df.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        if len(r_pot) == 0: r_pot = rentals_df.iloc[:, 0:5].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        r_data = r_pot.values.astype(int)
        R = len(r_data)

        # 4. Load Prices & Demand
        pri_data = get_clean_matrix('Prices')
        PRI_pg = {(p, g): pri_data[p, g] for p in range(pri_data.shape[0]) for g in range(G)}
        P = pri_data.shape[0]

        dem_raw = pd.read_excel(xls, 'Demand', header=None).values.flatten()
        dem_nums = [x for x in dem_raw if isinstance(x, (int, float)) and not np.isnan(x)]
        
        # --- STEP A: GREEDY PRICING ---
        print("  > Calculating Greedy Pricing...")
        pricing_policy_list = []
        chosen_prices = {} 
        DEM_rap = {}
        idx = 0
        max_idx = len(dem_nums)
        
        for r in range(R):
            g_idx = r_data[r, 4] - 1 
            for a in range(A + 1):
                best_rev = -1; best_p = 0
                for p in range(P):
                    dem = dem_nums[idx] if idx < max_idx else 0
                    DEM_rap[(r, a, p)] = dem
                    idx += 1
                    rev = dem * PRI_pg[(p, g_idx)]
                    if rev > best_rev: best_rev = rev; best_p = p
                chosen_prices[(r, a)] = best_p
                pricing_policy_list.append({'RentalID': r, 'Antecedence': a, 'PriceLevel': best_p})

        # --- STEP B: FLOW SIMULATION (IDEAL CAPACITY) ---
        print("  > Simulating Ideal Flow...")
        timelines = {} 
        
        for r in range(R):
            s_start = r_data[r, 0] - 1; s_end = r_data[r, 1] - 1
            t_start = r_data[r, 2]; t_end = r_data[r, 3]
            g_idx = r_data[r, 4] - 1
            
            for a in range(A + 1):
                p = chosen_prices[(r, a)]
                demand = DEM_rap.get((r, a, p), 0)
                
                if demand > 0:
                    key_start = (g_idx, s_start)
                    if key_start not in timelines: timelines[key_start] = []
                    timelines[key_start].append((t_start, -demand))
                    
                    if t_end <= T:
                        key_end = (g_idx, s_end)
                        if key_end not in timelines: timelines[key_end] = []
                        timelines[key_end].append((t_end, demand))

        # --- STEP C: CALCULATE NADIR & INITIAL COST ---
        w_owned_temp = {} # {(g,s): qty}
        total_cost = 0
        
        for (g, s), events in timelines.items():
            events.sort(key=lambda x: x[0])
            current_stock = 0; min_stock = 0
            for t, change in events:
                current_stock += change
                if current_stock < min_stock: min_stock = current_stock
            
            needed = abs(min_stock)
            if needed > 0:
                w_owned_temp[(g, s)] = needed
                total_cost += needed * COS_g[g]

        print(f"  > Ideal Cost: {total_cost:,.0f} | Budget: {BUD:,.0f}")

        # --- STEP D: BUDGET SCALING ---
        scale_factor = 1.0
        if total_cost > BUD:
            # Scale down to 95% of budget to be safe
            scale_factor = (BUD / total_cost) * 0.95
            print(f"  > Budget exceeded. Scaling fleet by factor: {scale_factor:.4f}")
        
        owned_capacity_list = []
        for (g, s), qty in w_owned_temp.items():
            # Apply scaling
            final_qty = math.floor(qty * scale_factor)
            if final_qty > 0:
                owned_capacity_list.append({'Group': g+1, 'Station': s+1, 'Quantity': final_qty})

        # Leased (Empty for start, let optimization handle it)
        leased_capacity_list = []

        # --- EXPORT ---
        print(f"  > Saving to {output_path}...")
        with pd.ExcelWriter(output_path) as writer:
            pd.DataFrame(owned_capacity_list).to_excel(writer, sheet_name='Owned_Capacity', index=False)
            pd.DataFrame(leased_capacity_list).to_excel(writer, sheet_name='Leased_Capacity', index=False)
            pd.DataFrame(pricing_policy_list).to_excel(writer, sheet_name='Pricing_Policy', index=False)
            
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# MAIN BATCH LOOP
# ==============================================================================
if __name__ == "__main__":
    
    DATA_FOLDER = "data"
    
    if os.path.exists(DATA_FOLDER):
        files = [f for f in os.listdir(DATA_FOLDER) if f.startswith("Inst") and f.endswith(".xlsx")]
        files.sort(key=lambda x: int(x.replace("Inst", "").replace(".xlsx", "")))
        
        print(f"Found {len(files)} instances.")
        for f in files:
            solve_budget_heuristic(os.path.join(DATA_FOLDER, f))
            
    else:
        print(f"Folder '{DATA_FOLDER}' not found.")