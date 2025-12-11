import os
import time
import math
import random
import copy
import gc
import pandas as pd
import numpy as np
from pyomo.environ import *

# ==============================================================================
# 1. SHARED CLASSES
# ==============================================================================
class InstanceData:
    def __init__(self, filename):
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
        def gp(n, t=float): v = pbg.loc[n].values.flatten(); return {g: t(v[g]) if g < len(v) else t(v[0]) for g in range(self.G)}
        self.LEA_g = gp('LEA_g'); self.LP_g = gp('LP_g', int); self.COS_g = gp('COS_g'); self.OWN_g = gp('OWN_g')

        pd_mat = get_mat('Prices')
        self.P = pd_mat.shape[0]
        self.PRI_pg = {(p, g): pd_mat[p, g] for p in range(self.P) for g in range(self.G)}

        dr = pd.read_excel(xls, 'Demand', header=None).values.flatten()
        dn = [x for x in dr if isinstance(x, (int, float)) and not np.isnan(x)]
        self.DEM_rap = {}; self.M_rap = {}
        idx = 0; max_idx = len(dn)
        for r in range(self.R):
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

class PersistentEvaluator:
    def __init__(self, data, is_relaxed=False):
        self.data = data
        self.model = ConcreteModel()
        m = self.model
        m.G = RangeSet(0, data.G-1); m.S = RangeSet(0, data.S-1); m.R = RangeSet(0, data.R-1)
        m.A = RangeSet(0, data.A); m.P = RangeSet(0, data.P-1); m.T = RangeSet(0, data.T); m.T_minus = RangeSet(0, data.T-1)

        domain_type = NonNegativeReals if is_relaxed else NonNegativeIntegers
        
        m.w_O = Var(m.G, m.S, domain=domain_type)
        m.w_L = Var(m.G, m.T_minus, m.S, domain=domain_type)
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

        def obj_rule(mod):
            rev = sum(mod.v[r, a, p] * data.PRI_pg[p, data.rental_gr[r]] for r in mod.R for a in mod.A for p in mod.P)
            cost_buy = sum(mod.w_O[g, s]*data.COS_g[g] for g in mod.G for s in mod.S)
            cost_lease = sum(mod.f_L[g, t]*data.LEA_g[g] for g in mod.G for t in mod.T_minus)
            cost_own = sum(mod.f_O[g, t]*data.OWN_g[g] for g in mod.G for t in mod.T_minus)
            cost_trans = sum((mod.y_L[s1, s2, g, t] + mod.y_O[s1, s2, g, t]) * data.TC_gs1s2[g, s1, s2] for s1 in mod.S for s2 in mod.S for g in mod.G for t in mod.T_minus)
            cost_upg = sum((mod.u_L[r, a, g]+mod.u_O[r, a, g])*data.PYU for g in mod.G for r in mod.R for a in mod.A if data.rental_gr[r] != g)
            return rev - cost_buy - cost_lease - cost_own - cost_trans - cost_upg
        m.obj = Objective(rule=obj_rule, sense=maximize)

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
        m.c_bud = Constraint(rule=lambda mod: sum(mod.w_O[g, s]*data.COS_g[g] for g in mod.G for s in mod.S) <= data.BUD)

        self.opt = SolverFactory('gurobi_persistent')
        self.opt.set_instance(self.model)
        self.opt.options['OutputFlag'] = 0 
        self.opt.options['MIPGap'] = 0.01

    def solve_for_pricing(self, pricing_policy):
        m = self.model
        data = self.data
        for r in m.R:
            for a in m.A:
                target = pricing_policy.get((r, a), -1)
                for p in m.P:
                    ub = data.DEM_rap[r, a, p] if p == target else 0
                    if m.v[r,a,p].ub != ub:
                        m.v[r,a,p].setub(ub)
                        self.opt.update_var(m.v[r,a,p])

        res = self.opt.solve(m, save_results=False, load_solutions=False, tee=False)
        
        if res.solver.termination_condition == TerminationCondition.optimal:
            self.opt.load_vars()
            current_owned = {}
            if m.w_O[0,0].domain == NonNegativeIntegers:
                for g in m.G:
                    for s in m.S:
                        if value(m.w_O[g,s]) > 0.5:
                            current_owned[(g,s)] = int(round(value(m.w_O[g,s])))
            
            sales_info = {} 
            if m.w_O[0,0].domain == NonNegativeReals:
                for r in m.R:
                    for a in m.A:
                        p_idx = pricing_policy.get((r,a), 0)
                        potential = data.DEM_rap[r, a, p_idx]
                        actual = value(m.v[r, a, p_idx])
                        sales_info[(r,a)] = (actual, potential)
            return value(m.obj), current_owned, sales_info
        else:
            return -1e9, {}, {}
    
    def close(self):
        try:
            if hasattr(self, 'opt') and self.opt:
                if hasattr(self.opt, '_solver_model'):
                    self.opt._solver_model.dispose()
        except Exception:
            pass

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def generate_randomized_greedy(data, randomness=0.3):
    target_p = {}
    for r in range(data.R):
        for a in range(data.A + 1):
            if random.random() < randomness:
                target_p[(r, a)] = random.randint(0, data.P - 1)
            else:
                best_rev = -1; best_p = 0
                for p in range(data.P):
                    rev = data.PRI_pg[p, data.rental_gr[r]] * data.DEM_rap[r, a, p]
                    if rev > best_rev: best_rev = rev; best_p = p
                target_p[(r, a)] = best_p
    return target_p

def saturation_guided_mutation(pricing_in, sales_info, data, aggressiveness=0.3):
    new_pricing = copy.deepcopy(pricing_in)
    
    if random.random() < 0.10:
        r_rand = random.randint(0, data.R - 1)
        a_rand = random.randint(0, data.A)
        new_pricing[(r_rand, a_rand)] = random.randint(0, data.P - 1)
        return new_pricing, True
    
    moves_made = False
    for (r, a), current_p in new_pricing.items():
        sales, potential = sales_info.get((r, a), (0, 0))
        if potential < 0.01: continue
        ratio = sales / potential
        
        if random.random() < aggressiveness:
            original_val = current_p
            if ratio > 0.9:
                if current_p < data.P - 1: new_pricing[(r, a)] = current_p + 1
            elif ratio > 0.1:
                if current_p < data.P - 1: new_pricing[(r, a)] = current_p + 1     
            else:
                if current_p > 0: new_pricing[(r, a)] = current_p - 1 
            
            if new_pricing[(r, a)] != original_val:
                moves_made = True
    return new_pricing, moves_made

# ==============================================================================
# 3. ALGORITHMS
# ==============================================================================
def run_sa_algorithm(data, evaluator_lp, initial_pricing, max_seconds=120):
    curr_price = copy.deepcopy(initial_pricing)
    curr_obj, _, curr_sales = evaluator_lp.solve_for_pricing(curr_price)
    best_obj_lp = curr_obj
    best_price = copy.deepcopy(curr_price)
    
    T = None; ALPHA = 0.99; aggressiveness = 0.2
    start_time = time.time()
    total_iter = 0
    last_print = 0
    accepts = 0
    rejects = 0
    improvements = 0
    
    # STOPPING CONDITION VARS
    no_improve_iters = 0
    MAX_NO_IMPROVE = 200 # Stop after 1000 iterations without finding a better solution
    
    print(f"\n  === SA STARTING ===")
    print(f"  Initial Obj: {curr_obj:,.2f} | Max Time: {max_seconds}s | Max Stagnant: {MAX_NO_IMPROVE}")
    print(f"  {'Iter':<6} {'Elapsed':<8} {'Curr_Obj':<15} {'New_Obj':<15} {'Delta':<12} {'Temp':<10} {'Accept':<12} {'Best':<15} {'Stagnant':<9} {'Aggr':<6}")
    print(f"  {'-'*130}")
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_seconds: 
            print(f"\n  SA STOPPED: Time limit reached ({elapsed:.1f}s)")
            break
        
        # Stop if stuck
        if no_improve_iters >= MAX_NO_IMPROVE:
            print(f"\n  SA STOPPED: No improvement for {no_improve_iters} iterations.")
            break

        total_iter += 1
        n_price, moved = saturation_guided_mutation(curr_price, curr_sales, data, aggressiveness)
        if not moved:
            r_rand = random.randint(0, data.R - 1); a_rand = random.randint(0, data.A)
            n_price[r_rand, a_rand] = random.randint(0, data.P - 1)

        n_obj, _, n_sales = evaluator_lp.solve_for_pricing(n_price)
        delta = n_obj - curr_obj
        
        accepted = False
        prob = 0.0
        if delta > 0: 
            accepted = True
            improvements += 1
        else:
            if T is None:
                base_T = delta / math.log(0.5) if delta < 0 else 0
                floor_T = abs(curr_obj) * 0.01
                T = max(base_T, floor_T)
                if T <= 1e-9: T = 1.0
                print(f"  Temperature initialized: T = {T:.2f}")
            try: prob = math.exp(delta / T)
            except: prob = 0
            if random.random() < prob: 
                accepted = True
        
        is_new_best = False
        if accepted:
            accepts += 1
            curr_price = n_price; curr_obj = n_obj; curr_sales = n_sales
            if curr_obj > best_obj_lp: 
                best_obj_lp = curr_obj
                best_price = copy.deepcopy(curr_price)
                no_improve_iters = 0 # Reset counter on improvement
                is_new_best = True
            else:
                no_improve_iters += 1
        else:
            rejects += 1
            no_improve_iters += 1 # Rejected move counts as no improvement
        
        # Print every iteration
        accept_str = "NEW BEST!" if is_new_best else ("Accept" if accepted else f"Reject p={prob:.4f}")
        temp_str = f"{T:.2f}" if T is not None else "N/A"
        print(f"  {total_iter:<6} {elapsed:<8.1f} {curr_obj:<15,.2f} {n_obj:<15,.2f} {delta:<12,.2f} {temp_str:<10} {accept_str:<12} {best_obj_lp:<15,.2f} {no_improve_iters:<9} {aggressiveness:<6.3f}")
        
        if T is not None: T = T * ALPHA
        if total_iter % 100 == 0: 
            aggressiveness = max(0.05, aggressiveness * 1)
            temp_display = f"{T:.2f}" if T is not None else "N/A"
            print(f"  --- Iter {total_iter}: Aggressiveness adjusted to {aggressiveness:.3f}, T={temp_display} ---")
        if T is not None and T < 1.0: 
            print(f"\n  SA STOPPED: Temperature cooled below 1.0 (T={T:.4f})")
            break 

    final_time = time.time() - start_time
    
    print(f"\n  === SA SUMMARY ===")
    print(f"  Total Iterations: {total_iter}")
    print(f"  Total Time: {final_time:.2f}s")
    print(f"  Accepts: {accepts} ({100*accepts/total_iter:.1f}%)")
    print(f"  Rejects: {rejects} ({100*rejects/total_iter:.1f}%)")
    print(f"  Improvements: {improvements} ({100*improvements/total_iter:.1f}%)")
    print(f"  Final Best Obj: {best_obj_lp:,.2f}")
    final_temp_display = f"{T:.4f}" if T is not None else "N/A"
    print(f"  Final Temp: {final_temp_display}")
    
    return best_obj_lp, total_iter, final_time

def run_ls_algorithm(data, evaluator_lp, initial_pricing, max_iters):
    curr_price = copy.deepcopy(initial_pricing)
    curr_obj, _, curr_sales = evaluator_lp.solve_for_pricing(curr_price)
    best_obj = curr_obj
    aggressiveness = 0.20
    improvements = 0
    
    print(f"\n  === LS STARTING ===")
    print(f"  Initial Obj: {curr_obj:,.2f} | Max Iterations: {max_iters}")
    print(f"  {'Iter':<6} {'Curr_Obj':<15} {'New_Obj':<15} {'Delta':<12} {'Status':<12} {'Best':<15} {'Improvements':<13} {'Aggr':<6}")
    print(f"  {'-'*110}")
    
    for i in range(max_iters):
        new_price, moved = saturation_guided_mutation(curr_price, curr_sales, data, aggressiveness)
        if not moved:
             r = random.randint(0, data.R - 1); a = random.randint(0, data.A)
             new_price[(r, a)] = random.randint(0, data.P - 1)
        new_obj, _, new_sales = evaluator_lp.solve_for_pricing(new_price)
        
        delta = new_obj - curr_obj
        status = ""
        
        if new_obj > curr_obj:
            improvements += 1
            curr_price = new_price; curr_obj = new_obj; curr_sales = new_sales
            if curr_obj > best_obj:
                best_obj = curr_obj
                status = "IMPROVED!"
            else:
                status = "Better"
        else:
            status = "Reject"
        
        # Print every iteration
        print(f"  {i+1:<6} {curr_obj:<15,.2f} {new_obj:<15,.2f} {delta:<12,.2f} {status:<12} {best_obj:<15,.2f} {improvements:<13} {aggressiveness:<6.3f}")
    
    print(f"\n  === LS SUMMARY ===")
    print(f"  Total Iterations: {max_iters}")
    print(f"  Improvements: {improvements} ({100*improvements/max_iters:.1f}%)")
    print(f"  Final Best Obj: {best_obj:,.2f}")
    print(f"  Gain from start: {best_obj - evaluator_lp.solve_for_pricing(initial_pricing)[0]:,.2f}")
            
    return curr_obj

def run_linear_with_gap(filename, time_limit):
    """
    Run the full MIP formulation (with pricing variables q[r,a,p])
    This is different from PersistentEvaluator which assumes fixed pricing.
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
    files += [f"data/Inst_Sim_{i}.xlsx" for i in range(1, 2)]
    # files += [f"data/Inst_Double_{i}.xlsx" for i in range(1, 6)]
    # files += [f"data/Inst_Quad_{i}.xlsx" for i in range(1, 3)]
    
    # files += [f"data/Inst{i}.xlsx" for i in range(1, 41)]
    
    OUTPUT_FILE = "Batch_Results_Full.xlsx"
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

    print(f"\n--- Starting FULL Batch Solve ({len(files)} files) ---")
    print("Algorithm: SA (600s or until stagnation) + LS + Greedy + Linear (300s)")

    for filename in files:
        if not os.path.exists(filename):
            print(f"Skipping {filename} (File Not Found)")
            continue
            
        if filename in completed_files:
            print(f"Skipping {filename} (Already Completed)")
            continue

        print(f"\nProcessing: {filename}")
        
        data = None
        eval_lp = None
        
        try:
            # --- SETUP ---
            data = InstanceData(filename)
            eval_lp = PersistentEvaluator(data, is_relaxed=True)
            
            # --- 1. CONSTRUCTIVE & GREEDY ---
            random.seed(42) 
            start_price = generate_randomized_greedy(data, randomness=0.00)
            start_obj, _, _ = eval_lp.solve_for_pricing(start_price)
            
            random.seed(42)
            greedy_price = generate_randomized_greedy(data, randomness=0.0)
            greedy_obj, _, _ = eval_lp.solve_for_pricing(greedy_price)
            
            print(f"  > Start Obj: {start_obj:,.0f} | Greedy Obj: {greedy_obj:,.0f}")

            # --- 2. SIMULATED ANNEALING ---
            print("  > Running SA (60s or stagnant)...")
            sa_obj, sa_iters, sa_time = run_sa_algorithm(
                data, eval_lp, start_price, max_seconds=600
            )
            print(f"  > SA Finished: {sa_obj:,.0f} (Iter: {sa_iters})")

            # --- 3. LOCAL SEARCH ---
            print(f"  > Running LS ({sa_iters} iters)...")
            ls_obj = run_ls_algorithm(
                data, eval_lp, start_price, max_iters=int(sa_iters)
            )
            print(f"  > LS Finished: {ls_obj:,.0f}")

            # --- 4. LINEAR SOLVER (WITH GAP) ---
            print("  > Running Linear Solver (300s)...")
            lin_obj, lin_gap = run_linear_with_gap(data, time_limit=300)
            print(f"  > Linear Finished: {lin_obj:,.0f} | Gap: {lin_gap}")

            # Collect results
            result = {
                'Instance': filename,
                'Start_Obj': start_obj,
                'Greedy_Obj': greedy_obj,
                'SA_Obj': sa_obj,
                'SA_Iters': sa_iters,
                'SA_Time': sa_time,
                'LS_Obj': ls_obj,
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
            if eval_lp: eval_lp.close(); del eval_lp
            if data: del data
            gc.collect()

    print(f"\nBatch Run Complete. Results saved to {OUTPUT_FILE}")