# ==============================================================================
# [M.EGI018] Operations Management Project
#
# SA - PRICING DRIVER (WITH RELAXATION CHECK & ADAPTIVE AGENT)
# ------------------------------------------------------------
# 1. Search Space: Pricing Policy ONLY.
# 2. Evaluator: 
#    - Calculates MIP (Integer Capacity) -> The Real Objective.
#    - Calculates LP (Relaxed Capacity)  -> The Bound/Benchmark.
# 3. Optimization: 
#    - Uses Reinforcement Learning (Adaptive Agent) to learn which prices work best.
# 4. Output: Compares Real vs Relaxed objective and time per iteration.
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
        self.G = int(up.loc['G'].iloc[0])
        self.S = int(up.loc['S'].iloc[0])
        self.A = int(up.loc['A'].iloc[0])
        self.T = int(up.loc['T'].iloc[0])
        self.PYU = float(up.loc['PYU'].iloc[0])
        self.BUD = float(up.loc['BUD'].iloc[0])

        rd = pd.read_excel(xls, 'RentalTypes')
        r_pot = rd.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        if len(r_pot) == 0: 
            r_pot = rd.iloc[:, 0:5].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        self.r_data = r_pot.values.astype(int)
        self.R = len(self.r_data)
        self.rental_gr = {r: self.r_data[r, 4] - 1 for r in range(self.R)}

        pbg = pd.read_excel(xls, 'ParametersByGroup', header=None, index_col=0)
        def gp(n, t=float): 
            v = pbg.loc[n].values.flatten()
            return {g: t(v[g]) if g < len(v) else t(v[0]) for g in range(self.G)}
        
        self.LEA_g = gp('LEA_g')
        self.LP_g = gp('LP_g', int)
        self.COS_g = gp('COS_g')
        self.OWN_g = gp('OWN_g')

        pd_mat = get_mat('Prices')
        self.P = pd_mat.shape[0]
        self.PRI_pg = {(p, g): pd_mat[p, g] for p in range(self.P) for g in range(self.G)}

        dr = pd.read_excel(xls, 'Demand', header=None).values.flatten()
        dn = [x for x in dr if isinstance(x, (int, float)) and not np.isnan(x)]
        self.DEM_rap = {}
        self.M_rap = {}
        idx = 0
        max_idx = len(dn)
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
                try: 
                    self.UPG_g1g2[(g1, g2)] = int(ud[g1, g2])
                except: 
                    self.UPG_g1g2[(g1, g2)] = 1 if g1 == g2 else 0

        tc = get_mat('TransferCosts')
        tt = get_mat('TransferTimes')
        self.TC_gs1s2 = {}
        self.TT_s1s2 = {}
        for s1 in range(self.S):
            for s2 in range(self.S):
                self.TT_s1s2[(s1, s2)] = int(tt[s1, s2])
                for g in range(self.G): 
                    self.TC_gs1s2[(g, s1, s2)] = float(tc[s1, s2]) if self.G == 1 or tc.shape == (self.S, self.S) else 0.0
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
        m.G = RangeSet(0, data.G-1)
        m.S = RangeSet(0, data.S-1)
        m.R = RangeSet(0, data.R-1)
        m.A = RangeSet(0, data.A)
        m.P = RangeSet(0, data.P-1)
        m.T = RangeSet(0, data.T)
        m.T_minus = RangeSet(0, data.T-1)

        # DECISION VARIABLES
        # If relaxed, use Reals. If not, use Integers.
        domain_type = NonNegativeReals if is_relaxed else NonNegativeIntegers
        
        m.w_O = Var(m.G, m.S, domain=domain_type)
        m.w_L = Var(m.G, m.T_minus, m.S, domain=domain_type)
        
        # Flow Variables (Always Relaxed for efficiency)
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

        # OBJECTIVE
        def obj_rule(mod):
            rev = sum(mod.v[r, a, p] * data.PRI_pg[p, data.rental_gr[r]] for r in mod.R for a in mod.A for p in mod.P)
            cost_buy = sum(mod.w_O[g, s] * data.COS_g[g] for g in mod.G for s in mod.S)
            cost_lease = sum(mod.f_L[g, t] * data.LEA_g[g] for g in mod.G for t in mod.T_minus)
            cost_own = sum(mod.f_O[g, t] * data.OWN_g[g] for g in mod.G for t in mod.T_minus)
            cost_trans = sum((mod.y_L[s1, s2, g, t] + mod.y_O[s1, s2, g, t]) * data.TC_gs1s2[g, s1, s2] for s1 in mod.S for s2 in mod.S for g in mod.G for t in mod.T_minus)
            cost_upg = sum((mod.u_L[r, a, g] + mod.u_O[r, a, g]) * data.PYU for g in mod.G for r in mod.R for a in mod.A if data.rental_gr[r] != g)
            return rev - cost_buy - cost_lease - cost_own - cost_trans - cost_upg
        m.obj = Objective(rule=obj_rule, sense=maximize)

        # CONSTRAINTS
        m.c_U = Constraint(m.R, m.A, rule=lambda mod, r, a: mod.U[r, a] == sum(mod.v[r, a, p] for p in mod.P))
        m.c1 = Constraint(m.R, m.A, rule=lambda mod, r, a: mod.U[r, a] == sum(mod.u_L[r, a, g] + mod.u_O[r, a, g] for g in mod.G))
        
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
             sum(mod.u_L[r, a, g] + mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 0] == s+1 and data.r_data[r, 2] == t) +
             sum(mod.y_L[s, s2, g, t] + mod.y_O[s, s2, g, t] for s2 in mod.S) <= mod.x_L[g, t, s] + mod.x_O[g, t, s])
        
        m.c_upg = Constraint(m.R, m.A, m.G, rule=lambda mod, r, a, g: 
                             mod.u_L[r, a, g] + mod.u_O[r, a, g] == 0 if data.UPG_g1g2.get((data.rental_gr[r], g), 0) == 0 and data.rental_gr[r] != g else Constraint.Skip)
        
        m.c_fL = Constraint(m.G, m.T, rule=lambda mod, g, t: mod.f_L[g, t] >= sum(mod.x_L[g, t, s] for s in mod.S) + sum(mod.u_L[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 2] <= t < data.r_data[r, 3]))
        m.c_fO = Constraint(m.G, m.T, rule=lambda mod, g, t: mod.f_O[g, t] >= sum(mod.x_O[g, t, s] for s in mod.S) + sum(mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 2] <= t < data.r_data[r, 3]))
        m.c_bud = Constraint(rule=lambda mod: sum(mod.w_O[g, s] * data.COS_g[g] for g in mod.G for s in mod.S) <= data.BUD)

        print("  Initializing Gurobi Persistent...")
        self.opt = SolverFactory('gurobi_persistent')
        self.opt.set_instance(self.model)
        self.opt.options['OutputFlag'] = 0 
        self.opt.options['MIPGap'] = 0.01

    def solve_for_pricing(self, pricing_policy):
        """
        Updates the bounds on 'v' based on the pricing policy and solves the MIP/LP.
        """
        m = self.model
        data = self.data

        # Update Pricing Logic (Bounds on v)
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
            
            # Extract optimal capacity (only needed for final save)
            current_owned = {}
            for g in m.G:
                for s in m.S:
                    if value(m.w_O[g,s]) > 0.5:
                        current_owned[(g,s)] = int(round(value(m.w_O[g,s])))
            
            return value(m.obj), current_owned
        else:
            return -1e9, {}

# ==============================================================================
# 3. HEURISTICS
# ==============================================================================
def generate_greedy_pricing(data):
    target_p = {}
    for r in range(data.R):
        for a in range(data.A + 1):
            best_rev = -1
            best_p = 0
            for p in range(data.P):
                rev = data.PRI_pg[p, data.rental_gr[r]] * data.DEM_rap[r, a, p]
                if rev > best_rev: 
                    best_rev = rev
                    best_p = p
            target_p[(r, a)] = best_p
    return target_p

# ==============================================================================
# 4. ADAPTIVE PRICING AGENT (REINFORCEMENT LEARNING)
# ==============================================================================
class AdaptivePricingAgent:
    def __init__(self, data, learning_rate=1.2, exploration_rate=0.3):
        self.data = data
        # Weight Matrix: Dictionary {(r, a): [weight_p0, weight_p1, ..., weight_P]}
        # Initially all prices have weight 1.0 (equal probability)
        self.weights = {}
        for r in range(data.R):
            for a in range(data.A + 1):
                self.weights[(r, a)] = [1.0] * data.P
        
        self.alpha = learning_rate       # How much we increase weight if result is good
        self.epsilon = exploration_rate  # Probability of choosing random (Exploration)

    def get_neighbor(self, current_pricing, mutation_rate=0.02):
        """
        Generates a neighbor using Learning (weights) or Randomness (epsilon).
        """
        new_pricing = copy.deepcopy(current_pricing)
        keys = list(new_pricing.keys())
        
        # Decide how many to change
        num_changes = max(1, int(len(keys) * mutation_rate))
        targets = random.sample(keys, num_changes)
        
        for (r, a) in targets:
            # Epsilon-Greedy Strategy:
            # If random < epsilon -> Explore (Random choice)
            # Else -> Exploit (Weighted choice based on memory)
            if random.random() < self.epsilon:
                new_p = random.randint(0, self.data.P - 1)
            else:
                w = self.weights[(r, a)]
                # Choose an index based on the weights
                new_p = random.choices(range(self.data.P), weights=w, k=1)[0]
            
            new_pricing[(r, a)] = new_p
            
        return new_pricing

    def learn(self, pricing_policy, quality_score):
        """
        Updates weights based on the result.
        quality_score: 0 = Rejected, 1 = Accepted, 2 = New Best
        """
        if quality_score <= 0:
            return 

        # If it's a new best, we learn faster (double reward)
        factor = self.alpha if quality_score == 1 else (self.alpha * 2)
        
        for (r, a), p_val in pricing_policy.items():
            # Increase the specific weight of the price used in this good solution
            self.weights[(r, a)][p_val] *= factor
            
            # Normalization to avoid huge numbers
            if self.weights[(r, a)][p_val] > 1000:
                total = sum(self.weights[(r, a)])
                self.weights[(r, a)] = [x/total for x in self.weights[(r, a)]]

def save_result(filename, w_owned_dict, pricing_dict):
    print(f"Saving to {filename}...")
    o_data = []
    p_data = []
    for (g, s), val in w_owned_dict.items():
        o_data.append({'Group': g+1, 'Station': s+1, 'Quantity': val})
    for (r, a), p in pricing_dict.items():
        p_data.append({'RentalID': r, 'Antecedence': a, 'PriceLevel': p})
    
    with pd.ExcelWriter(filename) as writer:
        pd.DataFrame(o_data).to_excel(writer, sheet_name='Owned_Capacity', index=False)
        pd.DataFrame(p_data).to_excel(writer, sheet_name='Pricing_Policy', index=False)

# ==============================================================================
# 5. SA MAIN LOOP (ADAPTIVE VERSION)
# ==============================================================================
def run_sa_pricing_driven(instance, max_seconds=120):
    print(f"Running SA (Adaptive Pricing + Relaxed Check) for {instance}")
    
    data = InstanceData(instance)
    
    # 1. Setup Evaluators
    evaluator_mip = PersistentEvaluator(data, is_relaxed=False) # Integer Model
    evaluator_lp  = PersistentEvaluator(data, is_relaxed=True)  # Relaxed Model (Benchmark)
    
    # 2. Setup Adaptive Agent (New)
    agent = AdaptivePricingAgent(data, learning_rate=1.2, exploration_rate=0.3)
    
    # 3. Initial Solution (Greedy Prices)
    curr_price = generate_greedy_pricing(data)
    
    print("Evaluating Initial Solution...")
    curr_obj, curr_w_o = evaluator_mip.solve_for_pricing(curr_price)
    print(f"Initial Profit: {curr_obj:,.0f}")
    
    best_obj = curr_obj
    best_price = copy.deepcopy(curr_price)
    best_w_o = copy.deepcopy(curr_w_o)
    
    # SA Params
    T = 5000
    alpha = 0.90
    start_time = time.time()
    total_iter = 0
    
    print("\nIter  | Temp       | New Obj         | Rel Obj         | Best Obj        | Eval(s)  | Status")
    print("-" * 110)
    
    while (time.time() - start_time) < max_seconds:
        total_iter += 1
        
        # 4. Perturb Pricing (USING THE AGENT NOW)
        # It chooses neighbors based on what it has learned so far
        n_price = agent.get_neighbor(curr_price, mutation_rate=0.02)
        
        # 5. Solve MIP (Real Objective)
        t0 = time.time()
        n_obj, n_w_o = evaluator_mip.solve_for_pricing(n_price)
        t1 = time.time()
        
        # 6. Solve LP (Relaxed Benchmark) - Optional, kept for comparison output
        n_obj_rel, _ = evaluator_lp.solve_for_pricing(n_price)
        
        time_mip = t1 - t0
        
        # 7. Acceptance Criteria
        delta = n_obj - curr_obj
        accepted = False
        learning_signal = 0 # 0=Nothing, 1=Accepted, 2=Best
        
        if delta > 0:
            accepted = True
            learning_signal = 1
        elif n_obj > -1e8: # Feasibility check
             try: 
                 prob = math.exp(delta / T)
             except: 
                 prob = 0
             if random.random() < prob: 
                 accepted = True
             
        status = ""
        if accepted:
            curr_price, curr_obj, curr_w_o = n_price, n_obj, n_w_o
            status = "ACCEPTED"
            
            if curr_obj > best_obj:
                best_obj, best_price, best_w_o = curr_obj, copy.deepcopy(curr_price), copy.deepcopy(curr_w_o)
                status = "NEW BEST"
                learning_signal = 2 # Strong reward
        
        # 8. LEARN (The Agent learns if the solution was good)
        if learning_signal > 0:
            agent.learn(n_price, learning_signal)
        
        print(f"{total_iter:<5} | {T:<10.2f} | {n_obj:<15,.0f} | {n_obj_rel:<15,.0f} | {best_obj:<15,.0f} | {time_mip:<8.4f} | {status}")
        
        # Cooling
        T *= alpha
        if T < 0.1: 
            T = 2000 # Reheat
        
    print(f"\nFinal Best Profit: {best_obj:,.0f}")
    save_result("BestSolution_SmartSA.xlsx", best_w_o, best_price)

if __name__ == "__main__":
    # Update this path to your specific file location
    INSTANCE = r"data\Inst41.xlsx"
    
    if os.path.exists(INSTANCE):
        run_sa_pricing_driven(INSTANCE, max_seconds=1200)
    else:
        print(f"File not found: {INSTANCE}")