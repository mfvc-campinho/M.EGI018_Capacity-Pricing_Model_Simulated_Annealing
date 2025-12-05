# ==============================================================================
"""
[M.EGI018] Operations Management Project

SA - SATURATION GUIDED + PROPER ANNEALING (METROPOLIS)
------------------------------------------------------
1. Mutation: "Saturation-Guided" (Yield Management Logic)
   - Targets Sold-Out rentals to RAISE price.
   - Targets Zero-Sales rentals to LOWER price.

2. Acceptance: "Metropolis Criterion"
   - Accepts WORSE solutions with probability exp(delta / T).
   - Allows escaping local optima to find better global peaks.
"""
# ==============================================================================
import os
import sys
import time
import math
import random
import copy
import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
from collections import deque

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

# ==============================================================================
# 2. PERSISTENT EVALUATOR
# ==============================================================================
class PersistentEvaluator:
    def __init__(self, data, is_relaxed=False):
        type_str = "RELAXED (LP)" if is_relaxed else "INTEGER (MIP)"
        print(f"Building Persistent Model ({type_str})...")
        
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

        print("  Initializing Gurobi Persistent...")
        self.opt = SolverFactory('gurobi_persistent')
        self.opt.set_instance(self.model)
        self.opt.options['OutputFlag'] = 0 
        self.opt.options['MIPGap'] = 0.01

    def solve_for_pricing(self, pricing_policy):
        """
        Solves the model and returns (ObjValue, OwnedFleetDict, SalesData)
        """
        m = self.model
        data = self.data

        # Update Pricing Logic
        for r in m.R:
            for a in m.A:
                target = pricing_policy.get((r, a), -1)
                for p in m.P:
                    ub = data.DEM_rap[r, a, p] if p == target else 0
                    if m.v[r,a,p].ub != ub:
                        m.v[r,a,p].setub(ub)
                        self.opt.update_var(m.v[r,a,p])

        res = self.opt.solve(m, save_results=False, load_solutions=False)
        
        if res.solver.termination_condition == TerminationCondition.optimal:
            self.opt.load_vars()
            
            current_owned = {}
            if m.w_O[0,0].domain == NonNegativeIntegers:
                for g in m.G:
                    for s in m.S:
                        if value(m.w_O[g,s]) > 0.5:
                            current_owned[(g,s)] = int(round(value(m.w_O[g,s])))
            
            # Extract Sales Info for Saturation Guidance
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

# ==============================================================================
# 3. SATURATION GUIDED LOGIC (Tabu Removed)
# ==============================================================================
def saturation_guided_mutation(pricing_in, sales_info, data, aggressiveness=0.3):
    """
    Removed tabu_list parameters and checks.
    """
    new_pricing = copy.deepcopy(pricing_in)
    
    # 1. PURE CHAOS (Random Perturbation)
    if random.random() < 0.10:
        r_rand = random.randint(0, data.R - 1)
        a_rand = random.randint(0, data.A)
        
        new_pricing[(r_rand, a_rand)] = random.randint(0, data.P - 1)
        # We don't strictly need to track moves for Tabu anymore, 
        # but we return True/False or the list to indicate a move happened.
        return new_pricing, True
    
    moves_made = False

    # 2. SATURATION LOGIC
    for (r, a), current_p in new_pricing.items():
        # TABU CHECK REMOVED HERE

        sales, potential = sales_info.get((r, a), (0, 0))
        if potential < 0.01: continue
            
        ratio = sales / potential
        
        if random.random() < aggressiveness:
            original_val = current_p
            
            if ratio > 0.99:
                if current_p < data.P - 1:
                    new_pricing[(r, a)] = current_p + 1
            elif ratio > 0.01:
                if current_p < data.P - 1:
                    new_pricing[(r, a)] = current_p + 1     
            else:
                if current_p > 0:
                    new_pricing[(r, a)] = current_p - 1 
            
            if new_pricing[(r, a)] != original_val:
                moves_made = True

    return new_pricing, moves_made

def generate_randomized_greedy(data, randomness=0.3):
    """
    Standardized Constructive Heuristic.
    """
    print(f"Generating Initial Solution (Randomness: {randomness*100:.0f}%)...")
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

# ==============================================================================
# 4. MAIN LOOP (DYNAMIC INITIAL TEMP)
# ==============================================================================
def run_proper_sa(instance, max_seconds=120):
    random.seed(42)
    print(f"Running Saturation-Guided Proper SA for {instance}")
    
    data = InstanceData(instance)
    evaluator_lp = PersistentEvaluator(data, is_relaxed=True)
    evaluator_mip = PersistentEvaluator(data, is_relaxed=False)
    
    curr_price = generate_randomized_greedy(data, randomness=0.30)
    curr_obj, _, curr_sales = evaluator_lp.solve_for_pricing(curr_price)
    
    best_obj_lp = curr_obj
    best_price = copy.deepcopy(curr_price)
    
    # SA Parameters
    T = None 
    ALPHA = 0.99
    
    # SETTING UPDATE: Matches local_search.py aggressiveness now? 
    # Or keep it distinct? I left it at 0.2 as per original file.
    aggressiveness = 0.2

    # Removed: tabu_tenure and tabu_list definition
    
    start_time = time.time()
    total_iter = 0
    accepted_worse = 0
    
    print(f"\n{'Iter':<5} | {'Temp':<10} | {'New Profit':<14} | {'Best Profit':<14} | {'Action'}")
    print("-" * 80)
    
    while (time.time() - start_time) < max_seconds:
        total_iter += 1
        note_str = "-"
        
        # 1. MUTATION (Removed tabu_list argument)
        n_price, moved = saturation_guided_mutation(
            curr_price, curr_sales, data, 
            aggressiveness=aggressiveness
        )
        
        if not moved:
            r_rand = random.randint(0, data.R - 1)
            a_rand = random.randint(0, data.A)
            n_price[r_rand, a_rand] = random.randint(0, data.P - 1)
            note_str = "Forced"

        # 2. Solve RELAXED LP
        n_obj, _, n_sales = evaluator_lp.solve_for_pricing(n_price)
        
        # 3. METROPOLIS ACCEPTANCE
        delta = n_obj - curr_obj
        accepted = False
        
        # CASE A: Improvement (Always Accept)
        if delta > 0:
            accepted = True
            note_str = "Improved"
            
        # CASE B: Worse (Accept with Probability)
        else:
            if T is None:
                if delta == 0:
                    T = 1000 
                else:
                    T = delta / math.log(0.5)
                print(f"  [Auto-Temp] Initialized T={T:.1f} based on first drop delta={delta:.1f}")

            try:
                prob = math.exp(delta / T)
            except OverflowError:
                prob = 0
            
            if random.random() < prob:
                accepted = True
                accepted_worse += 1
                note_str = f"Jump ({prob:.2f})"
        
        if accepted:
            curr_price = n_price
            curr_obj = n_obj
            curr_sales = n_sales
            
            # Removed: Tabu append logic
            
            if curr_obj > best_obj_lp:
                best_obj_lp = curr_obj
                best_price = copy.deepcopy(curr_price)
                note_str = "NEW RECORD !!!"

        # 4. COOLING
        if T is not None:
            T = T * ALPHA
        
        if total_iter % 100 == 0:
            aggressiveness = max(0.05, aggressiveness * 1)

        if total_iter % 20 == 0 or "RECORD" in note_str: 
            t_str = f"{int(T)}" if T is not None else "Pending"
            print(f"{total_iter:<5} | {t_str:<10} | {n_obj:<14,.0f} | {best_obj_lp:<14,.0f} | {note_str}")
        
        if T is not None and T < 1.0:
            print("System Frozen.")
            break

    print(f"\nSearch Finished. Total Iters: {total_iter}. Worse Accepted: {accepted_worse}")
    print(f"Best Relaxed Profit: {best_obj_lp:,.0f}")
    
    print("\n--- FINAL VALIDATION (MIP) ---")
    final_obj_mip, final_fleet, _ = evaluator_mip.solve_for_pricing(best_price)
    print(f"Final Real (Integer) Profit: {final_obj_mip:,.0f}")

if __name__ == "__main__":
    INSTANCE = r"data\\Inst39.xlsx" 
    if os.path.exists(INSTANCE):
        run_proper_sa(INSTANCE, max_seconds=1200)
    else:
        print("Instance not found.")