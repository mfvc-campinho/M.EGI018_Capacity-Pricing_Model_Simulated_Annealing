# ==============================================================================
# CAR RENTAL FLEET MANAGEMENT - LINEAR MODEL SOLVER
# Fixed: 
#  1. Transfer Costs now read correctly for G > 1
#  2. Upgrade Costs pre-calculated (proportional) in data reading
# ==============================================================================
import os
import sys
from pyomo.environ import *
import pandas as pd
import numpy as np

def solve_instance(filename):
    print(f"\n{'='*60}")
    print(f"PROCESSING: {filename}")
    print(f"{'='*60}")

    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found.")
        return {'Instance': filename, 'Status': 'File Not Found', 'Objective': None, 'Gap': None}

    try:
        xls = pd.ExcelFile(filename)

        def get_clean_matrix(sheet_name):
            df = pd.read_excel(xls, sheet_name, header=None)
            df_num = df.apply(pd.to_numeric, errors='coerce')
            return df_num.dropna(how='all', axis=0).dropna(how='all', axis=1).values

        # --- 1. Unit Parameters ---
        up_df = pd.read_excel(xls, 'UnitParameters', header=None, index_col=0)
        # Robust index handling
        try:
            G = int(up_df.loc['G'].iloc[0])
        except:
            # Fallback if index is not labeled
            G = int(up_df.iloc[0, 0])
            
        # Helper to safely get param by name or index
        def get_up_val(name, idx):
            if name in up_df.index: return float(up_df.loc[name].iloc[0])
            else: return float(up_df.iloc[idx, 0])

        S = int(get_up_val('S', 1))
        A = int(get_up_val('A', 2))
        T = int(get_up_val('T', 3))
        PYU = float(get_up_val('PYU', 4))
        BUD = float(get_up_val('BUD', 5))

        # --- 2. Rental Types ---
        rentals_df = pd.read_excel(xls, 'RentalTypes')
        r_pot = rentals_df.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        if len(r_pot) == 0:
            r_pot = rentals_df.iloc[:, 0:5].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        r_data = r_pot.values.astype(int)
        R = len(r_data)

        # --- 3. Parameters By Group ---
        pbg_df = pd.read_excel(xls, 'ParametersByGroup', header=None, index_col=0)
        def get_gp(name, dtype=float):
            vals = pbg_df.loc[name].values.flatten()
            vals = list(vals)
            if len(vals) < G:
                vals.extend([vals[-1]] * (G - len(vals)))
            return {g: dtype(vals[g]) for g in range(G)}

        LEA_g = get_gp('LEA_g')
        LP_g = get_gp('LP_g', int)
        COS_g = get_gp('COS_g')
        OWN_g = get_gp('OWN_g')

        # --- 4. Prices ---
        pri_data = get_clean_matrix('Prices')
        # If headers exist in the matrix, skip them. Inst40 usually has data from row 0 or 1
        # We assume the matrix is just numbers. If shape > G, we crop or map.
        PRI_pg = {}
        for p in range(pri_data.shape[0]):
            for g in range(G):
                val = pri_data[p, g] if g < pri_data.shape[1] else pri_data[p, -1]
                PRI_pg[(p, g)] = val
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

        # --- 6. Upgrades (Admissibility) ---
        upg_data = get_clean_matrix('Upgrades')
        UPG_g1g2 = {(g1, g2): int(upg_data[g1, g2]) if upg_data.size >= G*G else (1 if g1 == g2 else 0)
                    for g1 in range(G) for g2 in range(G)}

        # --- 7. Transfer Costs (FIXED READING LOGIC) ---
        tc_data = get_clean_matrix('TransferCosts')
        tt_data = get_clean_matrix('TransferTimes')
        
        TT_s1s2 = {(s1, s2): int(tt_data[s1, s2]) for s1 in range(S) for s2 in range(S)}
        TC_gs1s2 = {}
        
        # Check if we have a valid S x S matrix
        has_valid_tc = (tc_data.shape[0] >= S and tc_data.shape[1] >= S)
        
        for g in range(G):
            for s1 in range(S):
                for s2 in range(S):
                    if has_valid_tc:
                        TC_gs1s2[(g, s1, s2)] = float(tc_data[s1, s2])
                    else:
                        TC_gs1s2[(g, s1, s2)] = 0.0 # Only 0 if file is missing/wrong shape

        # Lookups
        rental_gr = {r: r_data[r, 4] - 1 for r in range(R)}
        INX_gs = {(g, s): 0.0 for g in range(G) for s in range(S)}
        
        # --- NEW: Pre-calculate Upgrade Costs ---
        # Cost = Levels * PYU. Only applicable if UPG_g1g2 == 1
        UPG_COST = {}
        for r in range(R):
            original_g = rental_gr[r]
            for target_g in range(G):
                if target_g >= original_g and UPG_g1g2.get((original_g, target_g), 0) == 1:
                    # Proportional Cost
                    UPG_COST[(r, target_g)] = (target_g - original_g) * PYU
                else:
                    UPG_COST[(r, target_g)] = 0.0

        # Param defaults
        ONY_gts = {(g, t, s, 'L'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)}
        ONY_gts.update({(g, t, s, 'O'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)})
        ONU_gts = {(g, t, s, 'L'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)}
        ONU_gts.update({(g, t, s, 'O'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)})

        # =================MODEL=================
        model = ConcreteModel()
        model.G = RangeSet(0, G-1)
        model.S = RangeSet(0, S-1)
        model.R = RangeSet(0, R-1)
        model.A = RangeSet(0, A)
        model.P = RangeSet(0, P-1)
        model.T = RangeSet(0, T)
        model.T_minus = RangeSet(0, T-1)

        # Variables
        model.w_O = Var(model.G, model.S, domain=NonNegativeIntegers)
        model.w_L = Var(model.G, model.T_minus, model.S, domain=NonNegativeIntegers)
        model.q = Var(model.R, model.A, model.P, domain=Binary)
        model.x_L = Var(model.G, model.T, model.S, domain=NonNegativeReals)
        model.x_O = Var(model.G, model.T, model.S, domain=NonNegativeReals)
        model.y_L = Var(model.S, model.S, model.G, model.T_minus, domain=NonNegativeIntegers)
        model.y_O = Var(model.S, model.S, model.G, model.T_minus, domain=NonNegativeIntegers)
        model.u_L = Var(model.R, model.A, model.G, domain=NonNegativeIntegers)
        model.u_O = Var(model.R, model.A, model.G, domain=NonNegativeIntegers)
        model.f_L = Var(model.G, model.T, domain=NonNegativeIntegers)
        model.f_O = Var(model.G, model.T, domain=NonNegativeIntegers)
        model.U = Var(model.R, model.A, domain=NonNegativeIntegers)
        model.v = Var(model.R, model.A, model.P, domain=NonNegativeIntegers)

        # --- OBJECTIVE ---
        model.obj = Objective(sense=maximize, rule=lambda m:
            sum(m.v[r, a, p] * PRI_pg[p, rental_gr[r]] for r in m.R for a in m.A for p in m.P) -
            (sum(m.w_O[g, s]*COS_g[g] for g in m.G for s in m.S) +
             sum(m.f_L[g, t]*LEA_g[g] for g in m.G for t in m.T_minus) +
             sum(m.f_O[g, t]*OWN_g[g] for g in m.G for t in m.T_minus) +
             sum((m.y_L[s1, s2, g, t] + m.y_O[s1, s2, g, t]) * TC_gs1s2[g, s1, s2] for s1 in m.S for s2 in m.S for g in m.G for t in m.T_minus) +
             
             # USE PRE-CALCULATED COST PARAMETER
             sum((m.u_L[r, a, g] + m.u_O[r, a, g]) * UPG_COST[(r, g)] 
                 for g in m.G for r in m.R for a in m.A if g > rental_gr[r])
            ))

        # Constraints
        model.c1 = Constraint(model.R, model.A, rule=lambda m, r, a: m.U[r, a] == sum(m.u_L[r, a, g]+m.u_O[r, a, g] for g in m.G))
        model.c2 = Constraint(model.R, model.A, model.P, rule=lambda m, r, a, p: m.v[r, a, p] <= M_rap[r, a]*m.q[r, a, p])
        model.c3 = Constraint(model.R, model.A, model.P, rule=lambda m, r, a, p: m.v[r, a, p] <= m.U[r, a])
        model.c4 = Constraint(model.R, model.A, model.P, rule=lambda m, r, a, p: m.v[r, a, p] >= m.U[r, a] - M_rap[r, a]*(1-m.q[r, a, p]))

        # Fleet constraints (Stock Balance)
        def stock_O(m, g, t, s):
            if t == 0: return m.x_O[g, 0, s] == INX_gs[g, s] + m.w_O[g, s]
            rin = sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 1] == s+1 and r_data[r, 3] == t)
            rout = sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t)
            tin = sum(m.y_O[s2, s, g, t-TT_s1s2[s2, s]] for s2 in m.S if t-TT_s1s2[s2, s] in m.T_minus)
            tout = sum(m.y_O[s, s2, g, t] for s2 in m.S if t in m.T_minus)
            return m.x_O[g, t, s] == m.x_O[g, t-1, s] + ONY_gts[g, t, s, 'O'] + ONU_gts[g, t, s, 'O'] + rin - rout + tin - tout
        model.c_stockO = Constraint(model.G, model.T, model.S, rule=stock_O)

        def stock_L(m, g, t, s):
            if t == 0: return m.x_L[g, 0, s] == 0
            rin = sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 1] == s+1 and r_data[r, 3] == t)
            rout = sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t)
            tin = sum(m.y_L[s2, s, g, t-TT_s1s2[s2, s]] for s2 in m.S if t-TT_s1s2[s2, s] in m.T_minus)
            tout = sum(m.y_L[s, s2, g, t] for s2 in m.S if t in m.T_minus)
            acq = m.w_L[g, t-1, s] if (t-1) in m.T_minus else 0
            if t <= LP_g[g]:
                return m.x_L[g, t, s] == (m.x_L[g, t-1, s] + ONY_gts[g, t, s, 'L'] + ONU_gts[g, t, s, 'L'] + acq + rin - rout + tin - tout)
            else:
                ret_idx = t - LP_g[g] - 1
                ret = m.w_L[g, ret_idx, s] if ret_idx in m.T_minus else 0
                return m.x_L[g, t, s] == (m.x_L[g, t-1, s] + ONY_gts[g, t, s, 'L'] + ONU_gts[g, t, s, 'L'] + acq - ret + rin - rout + tin - tout)
        model.c_stockL = Constraint(model.G, model.T, model.S, rule=stock_L)

        # Capacity, Demand, Budget
        model.c_cap = Constraint(model.G, model.T_minus, model.S, rule=lambda m, g, t, s:
             sum(m.u_L[r, a, g]+m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t) +
             sum(m.y_L[s, s2, g, t]+m.y_O[s, s2, g, t] for s2 in m.S) <= m.x_L[g, t, s] + m.x_O[g, t, s])

        model.c_dem = Constraint(model.R, model.A, model.P, rule=lambda m, r, a, p: sum(m.u_L[r, a, g]+m.u_O[r, a, g] for g in m.G) <= DEM_rap[r, a, p] + (1-m.q[r, a, p])*M_rap[r, a])
        model.c_price = Constraint(model.R, model.A, rule=lambda m, r, a: sum(m.q[r, a, p] for p in m.P) == 1)
        model.c_bud = Constraint(rule=lambda m: sum(m.w_O[g, s]*COS_g[g] for g in m.G for s in m.S) <= BUD)
        
        # Upgrade validity constraint
        model.c_upg = Constraint(model.R, model.A, model.G, rule=lambda m, r, a, g: 
            m.u_L[r, a, g]+m.u_O[r, a, g] == 0 if UPG_g1g2.get((rental_gr[r], g), 0) == 0 and rental_gr[r] != g else Constraint.Skip)

        model.c_fL = Constraint(model.G, model.T, rule=lambda m, g, t: m.f_L[g, t] >= sum(m.x_L[g, t, s] for s in m.S) + sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 2] <= t < r_data[r, 3]))
        model.c_fO = Constraint(model.G, model.T, rule=lambda m, g, t: m.f_O[g, t] >= sum(m.x_O[g, t, s] for s in m.S) + sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 2] <= t < r_data[r, 3]))

        # --- SOLVE ---
        print(f"\n[Gurobi Output for {filename} follows...]")
        opt = SolverFactory('gurobi')
        opt.options['TimeLimit'] = 600  
        opt.options['MIPGap'] = 0.01

        res = opt.solve(model, tee=True)

        stat = str(res.solver.termination_condition)
        try:
            obj_val = value(model.obj)
            try:
                lb = res.problem.lower_bound
                ub = res.problem.upper_bound
                if ub == float('inf') or ub == float('-inf') or lb == 0:
                    gap_str = "Unknown"
                else:
                    gap_val = abs((ub - lb) / lb)
                    gap_str = f"{gap_val*100:.2f}%"
            except:
                gap_str = "0.00%" if stat == 'optimal' else "Unknown"
        except Exception as e:
            obj_val = "N/A"
            gap_str = "Error"

        print(f"\n>>> FINISHED {filename} | Obj: {obj_val} | Gap: {gap_str}")
        return {'Instance': filename, 'Status': stat, 'Objective': obj_val, 'Gap': gap_str}

    except Exception as e:
        print(f"\n>>> FAILED {filename} | Error: {str(e)}")
        return {'Instance': filename, 'Status': 'Error', 'Objective': str(e), 'Gap': 'N/A'}