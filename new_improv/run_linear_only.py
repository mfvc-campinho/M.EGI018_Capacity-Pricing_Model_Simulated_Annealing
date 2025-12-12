import os
import time
import gc
import pandas as pd
import numpy as np
from pyomo.environ import *

# ==============================================================================
# LINEAR SOLVER WITH GAP TRACKING
# ==============================================================================
def run_linear_with_gap(filename, time_limit):
    """
    Run the full MIP formulation (with pricing variables q[r,a,p])
    """
    try:
        xls = pd.ExcelFile(filename)
        def get_clean_matrix(sheet_name):
            df = pd.read_excel(xls, sheet_name, header=None)
            df_num = df.apply(pd.to_numeric, errors='coerce')
            return df_num.dropna(how='all', axis=0).dropna(how='all', axis=1).values

        up_df = pd.read_excel(xls, 'UnitParameters', header=None, index_col=0)
        G = int(up_df.loc['G'].iloc[0]); S = int(up_df.loc['S'].iloc[0])
        A = int(up_df.loc['A'].iloc[0]); T = int(up_df.loc['T'].iloc[0])
        PYU = float(up_df.loc['PYU'].iloc[0]); BUD = float(up_df.loc['BUD'].iloc[0])

        rentals_df = pd.read_excel(xls, 'RentalTypes')
        r_pot = rentals_df.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        if len(r_pot) == 0: r_pot = rentals_df.iloc[:, 0:5].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        r_data = r_pot.values.astype(int); R = len(r_data)

        pbg_df = pd.read_excel(xls, 'ParametersByGroup', header=None, index_col=0)
        def get_gp(name, dtype=float):
            vals = pbg_df.loc[name].values.flatten()
            vals = vals[:G] if len(vals) >= G else [vals[0]] * G
            return {g: dtype(vals[g]) for g in range(G)}
        LEA_g = get_gp('LEA_g'); LP_g = get_gp('LP_g', int); COS_g = get_gp('COS_g'); OWN_g = get_gp('OWN_g')

        pri_data = get_clean_matrix('Prices')
        PRI_pg = {(p, g): pri_data[p, g] for p in range(pri_data.shape[0]) for g in range(G)}
        P = pri_data.shape[0]

        dem_raw = pd.read_excel(xls, 'Demand', header=None).values.flatten()
        dem_nums = [x for x in dem_raw if isinstance(x, (int, float)) and not np.isnan(x)]
        req_size = R * (A + 1) * P
        if len(dem_nums) < req_size: dem_nums.extend([0] * (req_size - len(dem_nums)))
        DEM_rap_arr = np.array(dem_nums[:req_size]).reshape((R, A+1, P))
        DEM_rap = {(r, a, p): DEM_rap_arr[r, a, p] for r in range(R) for a in range(A+1) for p in range(P)}
        M_rap = {(r, a): np.max(DEM_rap_arr[r, a, :]) for r in range(R) for a in range(A+1)}

        upg_data = get_clean_matrix('Upgrades')
        UPG_g1g2 = {(g1, g2): int(upg_data[g1, g2]) if upg_data.size >= G*G else (1 if g1 == g2 else 0) for g1 in range(G) for g2 in range(G)}

        tc_data = get_clean_matrix('TransferCosts'); tt_data = get_clean_matrix('TransferTimes')
        TC_gs1s2 = {}; TT_s1s2 = {(s1, s2): int(tt_data[s1, s2]) for s1 in range(S) for s2 in range(S)}
        if G == 1 and tc_data.shape >= (S, S):
             for s1 in range(S):
                 for s2 in range(S): TC_gs1s2[(0, s1, s2)] = float(tc_data[s1, s2]) 
        else:
             for g in range(G):
                 for s1 in range(S):
                     for s2 in range(S): TC_gs1s2[(g, s1, s2)] = 0.0

        rental_gr = {r: r_data[r, 4] - 1 for r in range(R)}
        INX_gs = {(g, s): 0.0 for g in range(G) for s in range(S)}

        model = ConcreteModel()
        model.G = RangeSet(0, G-1); model.S = RangeSet(0, S-1); model.R = RangeSet(0, R-1)
        model.A = RangeSet(0, A); model.P = RangeSet(0, P-1); model.T = RangeSet(0, T); model.T_minus = RangeSet(0, T-1)

        model.w_O = Var(model.G, model.S, domain=NonNegativeIntegers)
        model.w_L = Var(model.G, model.T_minus, model.S, domain=NonNegativeIntegers)
        model.q = Var(model.R, model.A, model.P, domain=Binary)
        model.x_L = Var(model.G, model.T, model.S, domain=NonNegativeIntegers)
        model.x_O = Var(model.G, model.T, model.S, domain=NonNegativeIntegers)
        model.y_L = Var(model.S, model.S, model.G, model.T_minus, domain=NonNegativeIntegers)
        model.y_O = Var(model.S, model.S, model.G, model.T_minus, domain=NonNegativeIntegers)
        model.u_L = Var(model.R, model.A, model.G, domain=NonNegativeIntegers)
        model.u_O = Var(model.R, model.A, model.G, domain=NonNegativeIntegers)
        model.f_L = Var(model.G, model.T, domain=NonNegativeIntegers)
        model.f_O = Var(model.G, model.T, domain=NonNegativeIntegers)
        model.U = Var(model.R, model.A, domain=NonNegativeIntegers)
        model.v = Var(model.R, model.A, model.P, domain=NonNegativeIntegers)

        model.obj = Objective(sense=maximize, rule=lambda m:
                              sum(m.v[r, a, p] * PRI_pg[p, rental_gr[r]] for r in m.R for a in m.A for p in m.P) -
                              (sum(m.w_O[g, s]*COS_g[g] for g in m.G for s in m.S) +
                                  sum(m.f_L[g, t]*LEA_g[g] for g in m.G for t in m.T_minus) +
                                  sum(m.f_O[g, t]*OWN_g[g] for g in m.G for t in m.T_minus) +
                               sum((m.y_L[s1, s2, g, t] + m.y_O[s1, s2, g, t]) * TC_gs1s2[g, s1, s2] for s1 in m.S for s2 in m.S for g in m.G for t in m.T_minus) +
                                  sum((m.u_L[r, a, g]+m.u_O[r, a, g])*PYU for g in m.G for r in m.R for a in m.A if rental_gr[r] != g)))

        model.c1 = Constraint(model.R, model.A, rule=lambda m, r, a: m.U[r, a] == sum(m.u_L[r, a, g]+m.u_O[r, a, g] for g in m.G))
        model.c2 = Constraint(model.R, model.A, model.P, rule=lambda m, r, a, p: m.v[r, a, p] <= M_rap[r, a]*m.q[r, a, p])
        model.c3 = Constraint(model.R, model.A, model.P, rule=lambda m, r, a, p: m.v[r, a, p] <= m.U[r, a])
        model.c4 = Constraint(model.R, model.A, model.P, rule=lambda m, r, a, p: m.v[r, a, p] >= m.U[r, a] - M_rap[r, a]*(1-m.q[r, a, p]))

        def stock_O(m, g, t, s):
            if t == 0: return m.x_O[g, 0, s] == INX_gs[g, s] + m.w_O[g, s]
            rin = sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 1] == s+1 and r_data[r, 3] == t)
            rout = sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t)
            tin = sum(m.y_O[s2, s, g, t-TT_s1s2[s2, s]] for s2 in m.S if t-TT_s1s2[s2, s] in m.T_minus)
            tout = sum(m.y_O[s, s2, g, t] for s2 in m.S if t in m.T_minus)
            return m.x_O[g, t, s] == m.x_O[g, t-1, s] + rin - rout + tin - tout
        model.c_stockO = Constraint(model.G, model.T, model.S, rule=stock_O)

        def stock_L(m, g, t, s):
            if t == 0: return m.x_L[g, 0, s] == 0
            rin = sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 1] == s+1 and r_data[r, 3] == t)
            rout = sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t)
            tin = sum(m.y_L[s2, s, g, t-TT_s1s2[s2, s]] for s2 in m.S if t-TT_s1s2[s2, s] in m.T_minus)
            tout = sum(m.y_L[s, s2, g, t] for s2 in m.S if t in m.T_minus)
            acq = m.w_L[g, t-1, s] if (t-1) in m.T_minus else 0
            if t <= LP_g[g]:
                return m.x_L[g, t, s] == (m.x_L[g, t-1, s] + acq + rin - rout + tin - tout)
            else:
                ret_index = t - LP_g[g] - 1
                ret = m.w_L[g, ret_index, s] if ret_index in m.T_minus else 0
                return m.x_L[g, t, s] == (m.x_L[g, t-1, s] + acq - ret + rin - rout + tin - tout)
        model.c_stockL = Constraint(model.G, model.T, model.S, rule=stock_L)
        
        model.c_cap = Constraint(model.G, model.T_minus, model.S, rule=lambda m, g, t, s:
                                 sum(m.u_L[r, a, g]+m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t) +
                                 sum(m.y_L[s, s2, g, t]+m.y_O[s, s2, g, t] for s2 in m.S) <= m.x_L[g, t, s] + m.x_O[g, t, s])

        model.c_dem = Constraint(model.R, model.A, model.P, rule=lambda m, r, a, p: sum(m.u_L[r, a, g]+m.u_O[r, a, g] for g in m.G) <= DEM_rap[r, a, p] + (1-m.q[r, a, p])*M_rap[r, a])
        model.c_price = Constraint(model.R, model.A, rule=lambda m, r, a: sum(m.q[r, a, p] for p in m.P) == 1)
        model.c_bud = Constraint(rule=lambda m: sum(m.w_O[g, s]*COS_g[g] for g in m.G for s in m.S) <= BUD)
        model.c_upg = Constraint(model.R, model.A, model.G, rule=lambda m, r, a, g: m.u_L[r, a, g]+m.u_O[r, a, g] == 0 if UPG_g1g2.get((rental_gr[r], g), 0) == 0 and rental_gr[r] != g else Constraint.Skip)
        model.c_fL = Constraint(model.G, model.T, rule=lambda m, g, t: m.f_L[g, t] >= sum(m.x_L[g, t, s] for s in m.S) + sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 2] <= t < r_data[r, 3]))
        model.c_fO = Constraint(model.G, model.T, rule=lambda m, g, t: m.f_O[g, t] >= sum(m.x_O[g, t, s] for s in m.S) + sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 2] <= t < r_data[r, 3]))

        opt = SolverFactory('gurobi')
        opt.options['TimeLimit'] = time_limit
        opt.options['MIPGap'] = 0.00
        res = opt.solve(model, tee=True)
        
        obj_val = value(model.obj) if res.solver.termination_condition == TerminationCondition.optimal else 0
        
        gap_str = "N/A"
        try:
            lb = res.problem.lower_bound
            ub = res.problem.upper_bound
            if ub == float('inf') or ub == float('-inf'):
                 gap_str = "Inf"
            else:
                 if abs(lb) > 1e-6:
                     gap_val = abs(ub - lb) / abs(lb)
                     gap_str = f"{gap_val*100:.2f}%"
                 else:
                     gap_str = "0.00%"
        except:
             gap_str = "Unknown"

        return obj_val, gap_str

    except Exception as e:
        print(f"Linear Solver Failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, "Error"

# ==============================================================================
# MAIN BATCH RUNNER
# ==============================================================================
if __name__ == "__main__":
    
    files = []
    
    files += [f"data/Inst{i}.xlsx" for i in range(1, 41)]
    
    OUTPUT_FILE = "Batch_Results_Linear.xlsx"
    results = []
    completed_files = set()

    if os.path.exists(OUTPUT_FILE):
        try:
            print(f"Found existing results in {OUTPUT_FILE}. Reading...")
            existing_df = pd.read_excel(OUTPUT_FILE)
            results = existing_df.to_dict('records')
            
            if 'Linear_Gap' in existing_df.columns:
                completed_files = set(
                    existing_df[existing_df['Linear_Gap'].notna()]['Instance'].astype(str).tolist()
                )
                print(f"-> Resuming. Identified {len(completed_files)} completed instances.")
        except Exception as e:
            print(f"Warning: Could not read existing file ({e}). Starting fresh.")

    print(f"\n--- Starting Linear Batch Solve ({len(files)} files) ---")
    print("Algorithm: Linear MIP (300s time limit)")

    for filename in files:
        if not os.path.exists(filename):
            print(f"Skipping {filename} (File Not Found)")
            continue
            
        if filename in completed_files:
            print(f"Skipping {filename} (Already Completed)")
            continue

        print(f"\nProcessing: {filename}")
        
        try:
            # --- LINEAR SOLVER (WITH GAP) ---
            print("  > Running Linear Solver (300s)...")
            lin_obj, lin_gap = run_linear_with_gap(filename, time_limit=300)
            print(f"  > Linear Finished: {lin_obj:,.0f} | Gap: {lin_gap}")

            # Collect results
            result = {
                'Instance': filename,
                'Linear_Obj': lin_obj,
                'Linear_Gap': lin_gap
            }
            results.append(result)
            
            # Save immediately
            pd.DataFrame(results).to_excel(OUTPUT_FILE, index=False)
            print(f"  > Saved to {OUTPUT_FILE}")

        except Exception as e:
            print(f"  > FAILED {filename}: {e}")
            results.append({'Instance': filename, 'Error': str(e)})
            pd.DataFrame(results).to_excel(OUTPUT_FILE, index=False)
        
        finally:
            gc.collect()

    print(f"\nBatch Run Complete. Results saved to {OUTPUT_FILE}")
