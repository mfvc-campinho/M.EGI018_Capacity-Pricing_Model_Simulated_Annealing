# ==============================================================================
"""
CAR RENTAL FLEET MANAGEMENT - NON-LINEAR (MINLP) SOLVER
(Based on Oliveira et al., 2018)

This script solves the full, original non-linear model (MINLP) directly
using Gurobi.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
from pyomo.environ import *
import pandas as pd
import numpy as np

# ==============================================================================
# MAIN SOLVING FUNCTION
# ==============================================================================

def solve_nonlinear_instance(filename):
    print(f"\n{'='*80}")
    print(f"STARTING NON-LINEAR (MINLP) SOLVE FOR: {filename}")
    print(f"{'='*80}")

    # Check if file exists
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found.")
        return

    # ==========================================================================
    # --- STEP 1: LOAD DATA ---
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 1: LOADING DATA from {filename}")
    print(f"{'='*60}")
    
    try:
        xls = pd.ExcelFile(filename)

        # --- HELPER ---
        def get_clean_matrix(sheet_name):
            df = pd.read_excel(xls, sheet_name, header=None)
            df_num = df.apply(pd.to_numeric, errors='coerce')
            return df_num.dropna(how='all', axis=0).dropna(how='all', axis=1).values

        # --- 1. Unit Parameters ---
        up_df = pd.read_excel(xls, 'UnitParameters', header=None, index_col=0)
        up_df.index = up_df.index.astype(str).str.strip()
        G = int(up_df.loc['G'].iloc[0])
        S = int(up_df.loc['S'].iloc[0])
        A = int(up_df.loc['A'].iloc[0])
        T = int(up_df.loc['T'].iloc[0])
        PYU = float(up_df.loc['PYU'].iloc[0])
        BUD = float(up_df.loc['BUD'].iloc[0])

        # --- 2. Rental Types (Robust) ---
        rentals_df = pd.read_excel(xls, 'RentalTypes')
        r_pot = rentals_df.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        if len(r_pot) == 0:
            r_pot = rentals_df.iloc[:, 0:5].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        r_data = r_pot.values.astype(int)
        R = len(r_data)

        # --- 3. Parameters By Group ---
        pbg_df = pd.read_excel(xls, 'ParametersByGroup', header=None, index_col=0)
        pbg_df.index = pbg_df.index.astype(str).str.strip()
        def get_gp(name, dtype=float):
            vals = pbg_df.loc[name].values.flatten()
            vals = vals[:G] if len(vals) >= G else [vals[0]] * G
            return {g: dtype(vals[g]) for g in range(G)}
        LEA_g = get_gp('LEA_g')
        LP_g = get_gp('LP_g', int)
        COS_g = get_gp('COS_g')
        OWN_g = get_gp('OWN_g')

        # --- 4. Prices ---
        pri_data = get_clean_matrix('Prices')
        PRI_pg = {(p, g): pri_data[p, g] for p in range(pri_data.shape[0]) for g in range(G)}
        P = pri_data.shape[0]

        # --- 5. Demand ---
        dem_raw = pd.read_excel(xls, 'Demand', header=None).values.flatten()
        dem_nums = [x for x in dem_raw if isinstance(x, (int, float)) and not np.isnan(x)]
        req_size = R * (A + 1) * P
        if len(dem_nums) < req_size:
            dem_nums.extend([0] * (req_size - len(dem_nums)))
        DEM_rap_arr = np.array(dem_nums[:req_size]).reshape((R, A+1, P))
        DEM_rap = {(r, a, p): DEM_rap_arr[r, a, p] for r in range(R) for a in range(A+1) for p in range(P)}
        M_rap = {(r, a): np.max(DEM_rap_arr[r, a, :]) for r in range(R) for a in range(A+1)}

        # --- 6. Upgrades ---
        upg_data = get_clean_matrix('Upgrades')
        UPG_g1g2 = {(g1, g2): int(upg_data[g1, g2]) if upg_data.size >= G*G else (1 if g1 == g2 else 0)
                    for g1 in range(G) for g2 in range(G)}

        # --- 7. Transfer Costs ---
        tc_data = get_clean_matrix('TransferCosts')
        TC_gs1s2 = {}
        if G == 1 and tc_data.shape >= (S, S):
            for s1 in range(S):
                for s2 in range(S): TC_gs1s2[(0, s1, s2)] = float(tc_data[s1, s2]) 
        else:
            for g in range(G):
                for s1 in range(S):
                    for s2 in range(S): TC_gs1s2[(g, s1, s2)] = 0.0

        # --- 8. Transfer Times ---
        tt_data = get_clean_matrix('TransferTimes')
        TT_s1s2 = {(s1, s2): int(tt_data[s1, s2]) for s1 in range(S) for s2 in range(S)}

        # --- 9. Lookups & Initial Conditions ---
        rental_gr = {r: r_data[r, 4] - 1 for r in range(R)}
        INX_gs = {(g, s): 0.0 for g in range(G) for s in range(S)}
        
        # Add ONY and ONU, defaulting to 0 as in paper
        ONY_gts = {(g, t, s, 'L'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)}
        ONY_gts.update({(g, t, s, 'O'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)})
        ONU_gts = {(g, t, s, 'L'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)}
        ONU_gts.update({(g, t, s, 'O'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)})

        print("Data loaded successfully.")

    except Exception as e:
        print(f"\n>>> FAILED to load data | Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================================================
    # --- STEP 2: BUILD AND SOLVE THE NON-LINEAR (MINLP) MODEL ---
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 2: BUILDING AND SOLVING NON-LINEAR (MINLP) MODEL")
    print(f"{'='*60}")
    
    model_nl = ConcreteModel()

    # ----- SETS -----
    model_nl.G = RangeSet(0, G-1)
    model_nl.S = RangeSet(0, S-1)
    model_nl.R = RangeSet(0, R-1)
    model_nl.A = RangeSet(0, A)
    model_nl.P = RangeSet(0, P-1)
    model_nl.T = RangeSet(0, T)
    model_nl.T_minus = RangeSet(0, T-1)

    # ----- Decision Variables (from nonlinear.py) -----
    model_nl.w_O = Var(model_nl.G, model_nl.S, domain=NonNegativeIntegers)
    model_nl.w_L = Var(model_nl.G, model_nl.T_minus, model_nl.S, domain=NonNegativeIntegers)
    model_nl.q = Var(model_nl.R, model_nl.A, model_nl.P, domain=Binary)
    model_nl.x_L = Var(model_nl.G, model_nl.T, model_nl.S, domain=NonNegativeIntegers)
    model_nl.x_O = Var(model_nl.G, model_nl.T, model_nl.S, domain=NonNegativeIntegers)
    model_nl.y_L = Var(model_nl.S, model_nl.S, model_nl.G, model_nl.T_minus, domain=NonNegativeIntegers)
    model_nl.y_O = Var(model_nl.S, model_nl.S, model_nl.G, model_nl.T_minus, domain=NonNegativeIntegers)
    model_nl.u_L = Var(model_nl.R, model_nl.A, model_nl.G, domain=NonNegativeIntegers)
    model_nl.u_O = Var(model_nl.R, model_nl.A, model_nl.G, domain=NonNegativeIntegers)
    model_nl.f_L = Var(model_nl.G, model_nl.T, domain=NonNegativeIntegers)
    model_nl.f_O = Var(model_nl.G, model_nl.T, domain=NonNegativeIntegers)

    # ----- Objective Function (NON-LINEAR) -----
    print("Setting NON-LINEAR objective function...")
    def obj_rule_nl(m):
        # Revenue (NON-LINEAR: u × q × price)
        revenue = sum((m.u_L[r, a, g] + m.u_O[r, a, g]) * m.q[r, a, p] * PRI_pg[p, rental_gr[r]]
                      for r in m.R for a in m.A for g in m.G for p in m.P)
        
        # Costs
        buy_cost = sum(m.w_O[g, s] * COS_g[g] for g in m.G for s in m.S)
        lease_cost = sum(m.f_L[g, t] * LEA_g[g] for g in m.G for t in m.T_minus)
        own_cost = sum(m.f_O[g, t] * OWN_g[g] for g in m.G for t in m.T_minus)
        transfer_cost = sum((m.y_L[s1, s2, g, t] + m.y_O[s1, s2, g, t]) * TC_gs1s2[g, s1, s2]
                           for s1 in m.S for s2 in m.S for g in m.G for t in m.T_minus)
        upgrade_cost = sum((m.u_L[r, a, g] + m.u_O[r, a, g]) * PYU
                          for g in m.G for r in m.R for a in m.A
                          if rental_gr[r] != g)
        
        return revenue - buy_cost - lease_cost - own_cost - transfer_cost - upgrade_cost
    model_nl.obj = Objective(rule=obj_rule_nl, sense=maximize)
    
    # ----- Constraints (Corrected per Paper) -----
    
    # Stock constraint for Owned fleet (Eq. 2 and 5)
    def stock_O(m, g, t, s):
        if t == 0:
            return m.x_O[g, 0, s] == INX_gs[g, s] + m.w_O[g, s]
        rin = sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 1] == s+1 and r_data[r, 3] == t)
        rout = sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t)
        tin = sum(m.y_O[s2, s, g, t-TT_s1s2[s2, s]] for s2 in m.S if t-TT_s1s2[s2, s] in m.T_minus)
        tout = sum(m.y_O[s, s2, g, t] for s2 in m.S if t in m.T_minus)
        ony = ONY_gts[g, t, s, 'O']
        onu = ONU_gts[g, t, s, 'O']
        return m.x_O[g, t, s] == m.x_O[g, t-1, s] + ony + onu + rin - rout + tin - tout
    model_nl.c_stockO = Constraint(model_nl.G, model_nl.T, model_nl.S, rule=stock_O)

    # Stock constraint for Leased fleet (Eq. 3, 4, and 6)
    def stock_L(m, g, t, s):
        if t == 0:
            return m.x_L[g, 0, s] == 0
        rin = sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 1] == s+1 and r_data[r, 3] == t)
        rout = sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t)
        tin = sum(m.y_L[s2, s, g, t-TT_s1s2[s2, s]] for s2 in m.S if t-TT_s1s2[s2, s] in m.T_minus)
        tout = sum(m.y_L[s, s2, g, t] for s2 in m.S if t in m.T_minus)
        acq = m.w_L[g, t-1, s] if (t-1) in m.T_minus else 0
        ony = ONY_gts[g, t, s, 'L']
        onu = ONU_gts[g, t, s, 'L']
        if t <= LP_g[g]:
            # Eq. (3)
            return m.x_L[g, t, s] == (m.x_L[g, t-1, s] + ony + onu + acq 
                                     + rin - rout + tin - tout)
        else:
            # Eq. (4)
            ret_index = t - LP_g[g] - 1
            ret = m.w_L[g, ret_index, s] if ret_index in m.T_minus else 0
            return m.x_L[g, t, s] == (m.x_L[g, t-1, s] + ony + onu + acq - ret 
                                     + rin - rout + tin - tout)
    model_nl.c_stockL = Constraint(model_nl.G, model_nl.T, model_nl.S, rule=stock_L)

    # Capacity constraint (Eq. 7)
    model_nl.c_cap = Constraint(model_nl.G, model_nl.T_minus, model_nl.S, rule=lambda m, g, t, s:
                             sum(m.u_L[r, a, g]+m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t) +
                             sum(m.y_L[s, s2, g, t]+m.y_O[s, s2, g, t] for s2 in m.S) <= m.x_L[g, t, s] + m.x_O[g, t, s])
    
    # Demand constraint (Eq. 8)
    model_nl.c_dem = Constraint(model_nl.R, model_nl.A, model_nl.P, rule=lambda m, r, a, p: sum(m.u_L[r, a, g]+m.u_O[r, a, g] for g in m.G) <= DEM_rap[r, a, p] + (1-m.q[r, a, p])*M_rap[r, a])
    
    # Price selection (Eq. 11)
    model_nl.c_price = Constraint(model_nl.R, model_nl.A, rule=lambda m, r, a: sum(m.q[r, a, p] for p in m.P) == 1)
    
    # Upgrade constraint (Eq. 9)
    model_nl.c_upg = Constraint(model_nl.R, model_nl.A, model_nl.G, rule=lambda m, r, a, g: m.u_L[r, a, g]+m.u_O[r, a, g] == 0 if UPG_g1g2.get((rental_gr[r], g), 0) == 0 and rental_gr[r] != g else Constraint.Skip)
    
    # Budget constraint (Eq. 10)
    model_nl.c_bud = Constraint(rule=lambda m: sum(m.w_O[g, s]*COS_g[g] for g in m.G for s in m.S) <= BUD)

    # Fleet size/costing constraints (Eq. 12)
    model_nl.c_fL = Constraint(model_nl.G, model_nl.T, rule=lambda m, g, t: m.f_L[g, t] >= sum(m.x_L[g, t, s] for s in m.S) + sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 2] <= t < r_data[r, 3]))
    model_nl.c_fO = Constraint(model_nl.G, model_nl.T, rule=lambda m, g, t: m.f_O[g, t] >= sum(m.x_O[g, t, s] for s in m.S) + sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 2] <= t < r_data[r, 3]))


    # ----- SOLVE NON-LINEAR MODEL -----
    print("\n[Gurobi output for NON-LINEAR model...]")
    solver_nl = SolverFactory('gurobi')
    
    # This tells Gurobi to solve a non-convex quadratic model (MINLP)
    solver_nl.options['NonConvex'] = 2 
    solver_nl.options['TimeLimit'] = 600 # 10 minute time limit
    solver_nl.options['MIPGap'] = 0.01  # 1% optimality gap
    
    res_nl = solver_nl.solve(model_nl, tee=True)

    # ==========================================================================
    # --- STEP 3: RESULTS ---
    # ==========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 3: FINAL RESULTS FOR {filename}")
    print(f"{'='*80}")

    print(f"  Solver Status: {res_nl.solver.status}")
    print(f"  Termination Condition: {res_nl.solver.termination_condition}")

    if res_nl.solver.termination_condition == TerminationCondition.optimal or \
       res_nl.solver.termination_condition == TerminationCondition.maxTimeLimit or \
       res_nl.solver.termination_condition == TerminationCondition.feasible:
        
        obj_val = value(model_nl.obj)
        print(f"\nObjective Value: {obj_val:.2f}")

        # Try to get gap
        try:
            lb = res_nl.problem.lower_bound
            ub = res_nl.problem.upper_bound
            gap = abs((ub - lb) / lb) if lb > 1e-6 else 0.0
            gap_str = f"{gap*100:.2f}%"
        except:
            gap_str = "0.00%" if str(res_nl.solver.termination_condition) == 'optimal' else "Unknown"

        print(f"Final Gap: {gap_str}")
        
        # --- (Optional) Print some solution details ---
        
        # FLEET PURCHASES (w_O, w_L)
        print(f"\n{'='*70}")
        print("FLEET PURCHASES")
        print(f"{'='*70}")
        total_owned = 0
        total_leased = 0
        
        print("\nOwned Fleet (w_O[g,s]):")
        for g in model_nl.G:
            for s in model_nl.S:
                val = value(model_nl.w_O[g, s])
                if val > 0.5:
                    print(f"  w_O[{g},{s}] = {val:.0f}")
                    total_owned += val
        print(f"Total owned fleet: {total_owned:.0f}")
        
        print("\nLeased Fleet (w_L[g,t,s]) - non-zero only:")
        for g in model_nl.G:
            for t in model_nl.T_minus:
                for s in model_nl.S:
                    val = value(model_nl.w_L[g, t, s])
                    if val > 0.5:
                        print(f"  w_L[{g},{t},{s}] = {val:.0f}")
                        total_leased += val
        print(f"Total leased fleet acquired: {total_leased:.0f}")

        # SUMMARY STATISTICS
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        
        total_rentals = sum(value(model_nl.u_L[r, a, g]) + value(model_nl.u_O[r, a, g])
                           for r in model_nl.R for a in model_nl.A for g in model_nl.G)
        print(f"Total rentals fulfilled: {total_rentals:.0f}")
        
        budget_used = sum(value(model_nl.w_O[g, s]) * COS_g[g] for g in model_nl.G for s in model_nl.S)
        print(f"Budget used: {budget_used:.2f} / {BUD:.2f}")

    else:
        print("\nNo feasible solution found.")


# ==============================================================================
# BATCH LOOP
# ==============================================================================
if __name__ == "__main__":
    
    # Run ONLY for Instance 1
    fname = r"data\Inst1.xlsx" # Use 'r' for raw string to handle backslash
    
    if not os.path.exists(fname):
        print(f"ERROR: Cannot find file at {fname}")
        print("Please make sure the 'data' folder is in the same directory as this script,")
        print("and it contains 'Inst1.xlsx'.")
    else:
        solve_nonlinear_instance(fname)

    print("\nBatch run finished.")