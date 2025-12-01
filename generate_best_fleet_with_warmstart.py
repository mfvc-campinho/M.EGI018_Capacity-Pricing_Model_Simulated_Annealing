# ==============================================================================
"""
[M.EGI018] Operations Management Project

INTEGRATED MILP SOLVER (MIP-HEURISTIC WARM START) - FIXED
---------------------------------------------------------
- Fixed 'NameError' by matching function names.
- Phase 1: Generates Constructive Heuristic (MIP).
- Phase 2: Injects it into Full MILP as Warm Start.
"""
# ==============================================================================
import os
import sys
import time
import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect

# ==============================================================================
# 1. DATA CLASS
# ==============================================================================
class InstanceData:
    def __init__(self, filename):
        print(f"Loading Instance Data from {filename}...")
        xls = pd.ExcelFile(filename)
        def get_mat(sheet): 
            return pd.read_excel(xls, sheet, header=None).apply(pd.to_numeric, errors='coerce').dropna(how='all').dropna(how='all', axis=1).values

        up = pd.read_excel(xls, 'UnitParameters', header=None, index_col=0)
        self.G = int(up.loc['G'].iloc[0]); self.S = int(up.loc['S'].iloc[0])
        self.A = int(up.loc['A'].iloc[0]); self.T = int(up.loc['T'].iloc[0])
        self.PYU = float(up.loc['PYU'].iloc[0]); self.BUD = float(up.loc['BUD'].iloc[0])

        rd = pd.read_excel(xls, 'RentalTypes')
        r_pot = rd.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        if len(r_pot)==0: r_pot = rd.iloc[:, 0:5].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        self.r_data = r_pot.values.astype(int)
        self.R = len(self.r_data)
        self.rental_gr = {r: self.r_data[r, 4] - 1 for r in range(self.R)}

        pbg = pd.read_excel(xls, 'ParametersByGroup', header=None, index_col=0)
        def gp(n, t=float): 
            v = pbg.loc[n].values.flatten()
            return {g: t(v[g]) if g < len(v) else t(v[0]) for g in range(self.G)}
        self.LEA_g = gp('LEA_g'); self.LP_g = gp('LP_g', int); self.COS_g = gp('COS_g'); self.OWN_g = gp('OWN_g')

        pd_mat = get_mat('Prices')
        self.P = pd_mat.shape[0]
        self.PRI_pg = {(p, g): pd_mat[p, g] for p in range(self.P) for g in range(self.G)}

        dr = pd.read_excel(xls, 'Demand', header=None).values.flatten()
        dn = [x for x in dr if isinstance(x, (int, float)) and not np.isnan(x)]
        self.DEM_rap = {}; self.M_rap = {}; self.demand_weights_gs = {}
        idx = 0; max_idx = len(dn)
        for r in range(self.R):
            s_start = self.r_data[r, 0] - 1; g_req = self.rental_gr[r]
            for a in range(self.A + 1):
                max_val = 0
                for p in range(self.P):
                    val = dn[idx] if idx < max_idx else 0
                    self.DEM_rap[(r, a, p)] = val
                    if val > max_val: max_val = val
                    idx += 1
                self.M_rap[(r, a)] = max_val

        ud = get_mat('Upgrades')
        self.UPG_g1g2 = {}
        for g1 in range(self.G):
            for g2 in range(self.G):
                try: self.UPG_g1g2[(g1, g2)] = int(ud[g1, g2])
                except: self.UPG_g1g2[(g1, g2)] = 1 if g1==g2 else 0

        tc = get_mat('TransferCosts'); tt = get_mat('TransferTimes')
        self.TC_gs1s2 = {}; self.TT_s1s2 = {}
        for s1 in range(self.S):
            for s2 in range(self.S):
                self.TT_s1s2[(s1, s2)] = int(tt[s1, s2])
                for g in range(self.G): 
                    self.TC_gs1s2[(g, s1, s2)] = float(tc[s1, s2]) if self.G==1 or tc.shape==(self.S,self.S) else 0.0
        self.INX_gs = {(g, s): 0.0 for g in range(self.G) for s in range(self.S)}
        
        self.ONY_gts = {(g, t, s, 'L'): 0.0 for g in range(self.G) for t in range(self.T+1) for s in range(self.S)}
        self.ONY_gts.update({(g, t, s, 'O'): 0.0 for g in range(self.G) for t in range(self.T+1) for s in range(self.S)})
        self.ONU_gts = {(g, t, s, 'L'): 0.0 for g in range(self.G) for t in range(self.T+1) for s in range(self.S)}
        self.ONU_gts.update({(g, t, s, 'O'): 0.0 for g in range(self.G) for t in range(self.T+1) for s in range(self.S)})

# ==============================================================================
# 2. PHASE 1: GENERATE WARM START (MIP HEURISTIC)
# ==============================================================================
def generate_mip_warm_start(data):
    print("\n--- PHASE 1: Generating Warm Start (MIP Heuristic) ---")
    
    # 1. Greedy Pricing
    target_p = {}
    pricing_policy_list = []
    for r in range(data.R):
        for a in range(data.A + 1):
            best_rev = -1; best_p = 0
            for p in range(data.P):
                rev = data.PRI_pg[p, data.rental_gr[r]] * data.DEM_rap[r, a, p]
                if rev > best_rev: best_rev = rev; best_p = p
            target_p[(r, a)] = best_p
            pricing_policy_list.append({'RentalID': r, 'Antecedence': a, 'PriceLevel': best_p})
    
    print("   > Greedy Prices Calculated.")
    print("   > Solving Capacity MIP (Budget Constrained)...")

    # 2. Build LP
    model = ConcreteModel()
    m = model
    m.G = RangeSet(0, data.G-1); m.S = RangeSet(0, data.S-1); m.R = RangeSet(0, data.R-1)
    m.A = RangeSet(0, data.A); m.P = RangeSet(0, data.P-1); m.T = RangeSet(0, data.T); m.T_minus = RangeSet(0, data.T-1)

    # INTEGERS for Physical Validity
    m.w_O = Var(m.G, m.S, domain=NonNegativeIntegers)
    m.w_L = Var(m.G, m.T_minus, m.S, domain=NonNegativeIntegers)
    
    # Relaxed Flow
    m.x_L = Var(m.G, m.T, m.S, domain=NonNegativeReals)
    m.x_O = Var(m.G, m.T, m.S, domain=NonNegativeReals)
    m.y_L = Var(m.S, m.S, m.G, m.T_minus, domain=NonNegativeReals)
    m.y_O = Var(m.S, m.S, m.G, m.T_minus, domain=NonNegativeReals)
    m.u_L = Var(m.R, m.A, m.G, domain=NonNegativeReals)
    m.u_O = Var(m.R, m.A, m.G, domain=NonNegativeReals)
    m.f_L = Var(m.G, m.T, domain=NonNegativeReals)
    m.f_O = Var(m.G, m.T, domain=NonNegativeReals)
    m.U = Var(m.R, m.A, domain=NonNegativeReals)
    m.v = Var(m.R, m.A, m.P, domain=NonNegativeReals)

    m.obj = Objective(sense=maximize, rule=lambda m:
        sum(m.v[r, a, p] * data.PRI_pg[p, data.rental_gr[r]] for r in m.R for a in m.A for p in m.P) -
        (sum(m.w_O[g, s]*data.COS_g[g] for g in m.G for s in m.S) +
         sum(m.f_L[g, t]*data.LEA_g[g] for g in m.G for t in m.T_minus) +
         sum(m.f_O[g, t]*data.OWN_g[g] for g in m.G for t in m.T_minus) +
         sum((m.y_L[s1, s2, g, t] + m.y_O[s1, s2, g, t]) * data.TC_gs1s2[g, s1, s2] for s1 in m.S for s2 in m.S for g in m.G for t in m.T_minus) +
         sum((m.u_L[r, a, g]+m.u_O[r, a, g])*data.PYU for g in m.G for r in m.R for a in m.A if data.rental_gr[r] != g)))

    # Fixed Price Constraint
    def v_rule(m, r, a, p):
        if p == target_p[(r, a)]: return m.v[r, a, p] <= data.DEM_rap[r, a, p]
        return m.v[r, a, p] == 0
    m.c_v = Constraint(m.R, m.A, m.P, rule=v_rule)
    
    m.c_bud = Constraint(rule=lambda m: sum(m.w_O[g, s]*data.COS_g[g] for g in m.G for s in m.S) <= data.BUD)

    m.c_U = Constraint(m.R, m.A, rule=lambda mod, r, a: mod.U[r, a] == sum(mod.v[r, a, p] for p in mod.P))
    m.c1 = Constraint(m.R, m.A, rule=lambda mod, r, a: mod.U[r, a] == sum(mod.u_L[r, a, g]+mod.u_O[r, a, g] for g in mod.G))
    
    def stock_O(mod, g, t, s):
        if t == 0: return mod.x_O[g, 0, s] == data.INX_gs[g, s] + mod.w_O[g, s]
        rin = sum(mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 1] == s+1 and data.r_data[r, 3] == t)
        rout = sum(mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 0] == s+1 and data.r_data[r, 2] == t)
        tin = sum(mod.y_O[s2, s, g, t-data.TT_s1s2[s2, s]] for s2 in mod.S if t-data.TT_s1s2[s2, s] in mod.T_minus)
        tout = sum(mod.y_O[s, s2, g, t] for s2 in mod.S if t in mod.T_minus)
        return mod.x_O[g, t, s] == mod.x_O[g, t-1, s] + rin - rout + tin - tout
    m.c_stockO = Constraint(m.G, m.T, m.S, rule=stock_O)

    def stock_L(mod, g, t, s):
        if t == 0: return mod.x_L[g, 0, s] == 0
        rin = sum(mod.u_L[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 1] == s+1 and data.r_data[r, 3] == t)
        rout = sum(mod.u_L[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 0] == s+1 and data.r_data[r, 2] == t)
        tin = sum(mod.y_L[s2, s, g, t-data.TT_s1s2[s2, s]] for s2 in mod.S if t-data.TT_s1s2[s2, s] in mod.T_minus)
        tout = sum(mod.y_L[s, s2, g, t] for s2 in mod.S if t in mod.T_minus)
        acq = mod.w_L[g, t-1, s] if (t-1) in mod.T_minus else 0
        if t <= data.LP_g[g]: return mod.x_L[g, t, s] == mod.x_L[g, t-1, s] + acq + rin - rout + tin - tout
        ret_idx = t - data.LP_g[g] - 1
        ret = mod.w_L[g, ret_idx, s] if ret_idx in mod.T_minus else 0
        return mod.x_L[g, t, s] == mod.x_L[g, t-1, s] + acq - ret + rin - rout + tin - tout
    m.c_stockL = Constraint(m.G, m.T, m.S, rule=stock_L)

    m.c_cap = Constraint(m.G, m.T_minus, m.S, rule=lambda mod, g, t, s:
             sum(mod.u_L[r, a, g]+mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 0] == s+1 and data.r_data[r, 2] == t) +
             sum(mod.y_L[s, s2, g, t]+mod.y_O[s, s2, g, t] for s2 in mod.S) <= mod.x_L[g, t, s] + mod.x_O[g, t, s])
    m.c_upg = Constraint(m.R, m.A, m.G, rule=lambda mod, r, a, g: 
                                 mod.u_L[r, a, g]+mod.u_O[r, a, g] == 0 if data.UPG_g1g2.get((data.rental_gr[r], g), 0) == 0 and data.rental_gr[r] != g else Constraint.Skip)
    m.c_fL = Constraint(m.G, m.T, rule=lambda mod, g, t: mod.f_L[g, t] >= sum(mod.x_L[g, t, s] for s in mod.S) + sum(mod.u_L[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 2] <= t < data.r_data[r, 3]))
    m.c_fO = Constraint(m.G, m.T, rule=lambda mod, g, t: mod.f_O[g, t] >= sum(mod.x_O[g, t, s] for s in mod.S) + sum(mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 2] <= t < data.r_data[r, 3]))

    # Solve
    opt = SolverFactory('gurobi')
    opt.options['TimeLimit'] = 60
    res = opt.solve(model, tee=True)

    w_owned_out = {}
    w_leased_out = {}
    
    if res.solver.termination_condition == TerminationCondition.optimal:
        print(f"   > MIP Optimal! Objective: {value(model.obj):,.2f}")
        
        o_data = []; l_data = []
        for g in m.G:
            for s in m.S:
                val = value(m.w_O[g, s])
                if val > 0.5: 
                    w_owned_out[(g, s)] = int(round(val)) 
                    o_data.append({'Group': g+1, 'Station': s+1, 'Quantity': int(round(val))})
        
        for g in m.G:
            for t in m.T_minus:
                for s in m.S:
                    val = value(m.w_L[g, t, s])
                    if val > 0.5: 
                        w_leased_out[(g, t, s)] = int(round(val))
                        l_data.append({'Group': g+1, 'Time': t, 'Station': s+1, 'Quantity': int(round(val))})
        
        with pd.ExcelWriter("Initial_Solution.xlsx") as writer:
            pd.DataFrame(o_data).to_excel(writer, sheet_name='Owned_Capacity', index=False)
            pd.DataFrame(l_data).to_excel(writer, sheet_name='Leased_Capacity', index=False)
            pd.DataFrame(pricing_policy_list).to_excel(writer, sheet_name='Pricing_Policy', index=False)
        print(f"   > Initial Solution saved to Initial_Solution.xlsx")
    else:
        print("   > MIP Heuristic Failed. Using empty capacity.")

    return w_owned_out, w_leased_out, target_p

# ==============================================================================
# 3. PHASE 2: FULL MILP SOLVER (WITH WARM START)
# ==============================================================================
def run_full_milp(data, w_o_init, w_l_init, price_init, time_limit=120):
    print("\n--- PHASE 2: FULL MILP OPTIMIZATION ---")
    
    model = ConcreteModel()
    m = model
    m.G = RangeSet(0, data.G-1); m.S = RangeSet(0, data.S-1); m.R = RangeSet(0, data.R-1)
    m.A = RangeSet(0, data.A); m.P = RangeSet(0, data.P-1); m.T = RangeSet(0, data.T); m.T_minus = RangeSet(0, data.T-1)

    m.w_O = Var(m.G, m.S, domain=NonNegativeIntegers)
    m.w_L = Var(m.G, m.T_minus, m.S, domain=NonNegativeIntegers)
    m.q = Var(m.R, m.A, m.P, domain=Binary)
    
    m.x_L = Var(m.G, m.T, m.S, domain=NonNegativeReals)
    m.x_O = Var(m.G, m.T, m.S, domain=NonNegativeReals)
    m.y_L = Var(m.S, m.S, m.G, m.T_minus, domain=NonNegativeReals)
    m.y_O = Var(m.S, m.S, m.G, m.T_minus, domain=NonNegativeReals)
    m.u_L = Var(m.R, m.A, m.G, domain=NonNegativeReals)
    m.u_O = Var(m.R, m.A, m.G, domain=NonNegativeReals)
    m.f_L = Var(m.G, m.T, domain=NonNegativeReals)
    m.f_O = Var(m.G, m.T, domain=NonNegativeReals)
    m.U = Var(m.R, m.A, domain=NonNegativeReals)
    m.v = Var(m.R, m.A, m.P, domain=NonNegativeReals)

    m.obj = Objective(sense=maximize, rule=lambda m:
        sum(m.v[r, a, p] * data.PRI_pg[p, data.rental_gr[r]] for r in m.R for a in m.A for p in m.P) -
        (sum(m.w_O[g, s]*data.COS_g[g] for g in m.G for s in m.S) +
         sum(m.f_L[g, t]*data.LEA_g[g] for g in m.G for t in m.T_minus) +
         sum(m.f_O[g, t]*data.OWN_g[g] for g in m.G for t in m.T_minus) +
         sum((m.y_L[s1, s2, g, t] + m.y_O[s1, s2, g, t]) * data.TC_gs1s2[g, s1, s2] for s1 in m.S for s2 in m.S for g in m.G for t in m.T_minus) +
         sum((m.u_L[r, a, g]+m.u_O[r, a, g])*data.PYU for g in m.G for r in m.R for a in m.A if data.rental_gr[r] != g)))

    m.c1 = Constraint(m.R, m.A, rule=lambda mod, r, a: mod.U[r, a] == sum(mod.u_L[r, a, g]+mod.u_O[r, a, g] for g in mod.G))
    m.c2 = Constraint(m.R, m.A, m.P, rule=lambda mod, r, a, p: mod.v[r, a, p] <= data.M_rap[r, a]*mod.q[r, a, p])
    m.c3 = Constraint(m.R, m.A, m.P, rule=lambda mod, r, a, p: mod.v[r, a, p] <= mod.U[r, a])
    m.c4 = Constraint(m.R, m.A, m.P, rule=lambda mod, r, a, p: mod.v[r, a, p] >= mod.U[r, a] - data.M_rap[r, a]*(1-mod.q[r, a, p]))
    
    def stock_O(mod, g, t, s):
        if t == 0: return mod.x_O[g, 0, s] == data.INX_gs[g, s] + mod.w_O[g, s]
        rin = sum(mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 1] == s+1 and data.r_data[r, 3] == t)
        rout = sum(mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 0] == s+1 and data.r_data[r, 2] == t)
        tin = sum(mod.y_O[s2, s, g, t-data.TT_s1s2[s2, s]] for s2 in mod.S if t-data.TT_s1s2[s2, s] in mod.T_minus)
        tout = sum(mod.y_O[s, s2, g, t] for s2 in mod.S if t in mod.T_minus)
        return mod.x_O[g, t, s] == mod.x_O[g, t-1, s] + rin - rout + tin - tout
    m.c_stockO = Constraint(m.G, m.T, m.S, rule=stock_O)
    
    def stock_L(mod, g, t, s):
        if t == 0: return mod.x_L[g, 0, s] == 0
        rin = sum(mod.u_L[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 1] == s+1 and data.r_data[r, 3] == t)
        rout = sum(mod.u_L[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 0] == s+1 and data.r_data[r, 2] == t)
        tin = sum(mod.y_L[s2, s, g, t-data.TT_s1s2[s2, s]] for s2 in mod.S if t-data.TT_s1s2[s2, s] in mod.T_minus)
        tout = sum(mod.y_L[s, s2, g, t] for s2 in mod.S if t in mod.T_minus)
        acq = mod.w_L[g, t-1, s] if (t-1) in mod.T_minus else 0
        if t <= data.LP_g[g]: return mod.x_L[g, t, s] == mod.x_L[g, t-1, s] + acq + rin - rout + tin - tout
        ret_idx = t - data.LP_g[g] - 1
        ret = mod.w_L[g, ret_idx, s] if ret_idx in mod.T_minus else 0
        return mod.x_L[g, t, s] == mod.x_L[g, t-1, s] + acq - ret + rin - rout + tin - tout
    m.c_stockL = Constraint(m.G, m.T, m.S, rule=stock_L)

    m.c_cap = Constraint(m.G, m.T_minus, m.S, rule=lambda mod, g, t, s:
         sum(mod.u_L[r, a, g]+mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 0] == s+1 and data.r_data[r, 2] == t) +
         sum(mod.y_L[s, s2, g, t]+mod.y_O[s, s2, g, t] for s2 in mod.S) <= mod.x_L[g, t, s] + mod.x_O[g, t, s])
    m.c_dem = Constraint(m.R, m.A, m.P, rule=lambda mod, r, a, p: sum(mod.u_L[r, a, g]+mod.u_O[r, a, g] for g in mod.G) <= data.DEM_rap[r, a, p] + (1-mod.q[r, a, p])*data.M_rap[r, a])
    m.c_price = Constraint(m.R, m.A, rule=lambda mod, r, a: sum(mod.q[r, a, p] for p in mod.P) == 1)
    m.c_bud = Constraint(rule=lambda mod: sum(mod.w_O[g, s]*data.COS_g[g] for g in mod.G for s in mod.S) <= data.BUD)
    m.c_upg = Constraint(m.R, m.A, m.G, rule=lambda mod, r, a, g: 
                             mod.u_L[r, a, g]+mod.u_O[r, a, g] == 0 if data.UPG_g1g2.get((data.rental_gr[r], g), 0) == 0 and data.rental_gr[r] != g else Constraint.Skip)
    m.c_fL = Constraint(m.G, m.T, rule=lambda mod, g, t: mod.f_L[g, t] >= sum(mod.x_L[g, t, s] for s in mod.S) + sum(mod.u_L[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 2] <= t < data.r_data[r, 3]))
    m.c_fO = Constraint(m.G, m.T, rule=lambda mod, g, t: mod.f_O[g, t] >= sum(mod.x_O[g, t, s] for s in mod.S) + sum(mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 2] <= t < data.r_data[r, 3]))

    # --- INJECT WARM START ---
    print("Injecting Warm Start...")
    # 1. Capacity
    for idx in w_o_init: m.w_O[idx].value = value(w_o_init[idx])
    for idx in w_l_init: m.w_L[idx].value = value(w_l_init[idx])
    
    # 2. Pricing (Set chosen to 1, others to 0)
    for r in m.R:
        for a in m.A:
            target = price_init[(r, a)]
            for p in m.P:
                if p == target: m.q[r, a, p].value = 1
                else: m.q[r, a, p].value = 0

    # --- SOLVE ---
    print(f"Starting Gurobi Solve (Limit: {time_limit}s)...")
    opt = SolverFactory('gurobi')
    opt.options['TimeLimit'] = time_limit
    opt.options['MIPGap'] = 0.01 
    
    res = opt.solve(model, tee=True, warmstart=True)

    if res.solver.termination_condition in ['optimal', 'maxTimeLimit', 'feasible']:
        print(f"\n>>> FINAL OBJECTIVE: {value(model.obj):,.2f}")
        
        # Export
        o_data = []; l_data = []; p_data = []
        for g in m.G:
            for s in m.S:
                if value(m.w_O[g,s]) > 0.5: o_data.append({'Group': g+1, 'Station': s+1, 'Quantity': int(value(m.w_O[g,s]))})
        for g in m.G:
            for t in m.T_minus:
                for s in m.S:
                    if value(m.w_L[g,t,s]) > 0.5: l_data.append({'Group': g+1, 'Time': t, 'Station': s+1, 'Quantity': int(value(m.w_L[g,t,s]))})
        for r in m.R:
            for a in m.A:
                for p in m.P:
                    if value(m.q[r,a,p]) > 0.5: 
                        p_data.append({'RentalID': r, 'Antecedence': a, 'PriceLevel': p})
                        break
        
        with pd.ExcelWriter("MILP_BestSolution_Inst41.xlsx") as writer:
            pd.DataFrame(o_data).to_excel(writer, sheet_name='Owned_Capacity', index=False)
            pd.DataFrame(l_data).to_excel(writer, sheet_name='Leased_Capacity', index=False)
            pd.DataFrame(p_data).to_excel(writer, sheet_name='Pricing_Policy', index=False)
        print("Saved to MILP_BestSolution_Inst41.xlsx")
    else:
        print("Optimization failed.")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    INSTANCE = r"data\Inst41.xlsx"
    
    if os.path.exists(INSTANCE):
        # 1. Load Data
        data = InstanceData(INSTANCE)
        
        # 2. Run Heuristic (Phase 1) - Now returns Integers
        w_o_init, w_l_init, price_init = generate_mip_warm_start(data)
        
        # 3. Run Full Optimization (Phase 2)
        if w_o_init:
            run_full_milp(data, w_o_init, w_l_init, price_init, time_limit=1200)
    else:
        print(f"Error: {INSTANCE} not found.")