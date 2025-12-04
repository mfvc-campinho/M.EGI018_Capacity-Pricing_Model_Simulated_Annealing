# ==============================================================================
"""
[M.EGI018] Operations Management Project

SA - FAST RELAXATION ACCELERATOR
--------------------------------
Agora usando:
- Excel heuristic_price_from_capacity_all_instances.xlsx para preços iniciais
- Solver Gurobi "normal" (não-persistente) para evitar erros de variáveis
  não inicializadas na interface persistent.
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


# ==============================================================================
# Helper: Lê preços iniciais do Excel
# ==============================================================================
def load_constructive_price_from_excel(excel_file, sheet_name):
    """
    Lê a aba sheet_name do Excel com colunas:
        rental_type, antecedence, p_init
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    required_cols = {"rental_type", "antecedence", "p_init"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"A aba '{sheet_name}' não contém as colunas {required_cols}"
        )
    return df


# ==============================================================================
# 1. DATA CLASS
# ==============================================================================
class InstanceData:
    def __init__(self, filename):
        print(f"Loading Instance Data from {filename}...")
        xls = pd.ExcelFile(filename)

        def get_mat(sheet):
            return (
                pd.read_excel(xls, sheet, header=None)
                .apply(pd.to_numeric, errors="coerce")
                .dropna(how="all")
                .dropna(how="all", axis=1)
                .values
            )

        up = pd.read_excel(xls, "UnitParameters", header=None, index_col=0)
        self.G = int(up.loc["G"].iloc[0])
        self.S = int(up.loc["S"].iloc[0])
        self.A = int(up.loc["A"].iloc[0])
        self.T = int(up.loc["T"].iloc[0])
        self.PYU = float(up.loc["PYU"].iloc[0])
        self.BUD = float(up.loc["BUD"].iloc[0])

        rd = pd.read_excel(xls, "RentalTypes")
        r_pot = rd.iloc[:, 1:6].apply(pd.to_numeric, errors="coerce").dropna(how="any")
        if len(r_pot) == 0:
            r_pot = (
                rd.iloc[:, 0:5]
                .apply(pd.to_numeric, errors="coerce")
                .dropna(how="any")
            )
        self.r_data = r_pot.values.astype(int)
        self.R = len(self.r_data)
        self.rental_gr = {r: self.r_data[r, 4] - 1 for r in range(self.R)}

        pbg = pd.read_excel(xls, "ParametersByGroup", header=None, index_col=0)

        def gp(n, t=float):
            v = pbg.loc[n].values.flatten()
            return {g: t(v[g]) if g < len(v) else t(v[0]) for g in range(self.G)}

        self.LEA_g = gp("LEA_g")
        self.LP_g = gp("LP_g", int)
        self.COS_g = gp("COS_g")
        self.OWN_g = gp("OWN_g")

        pd_mat = get_mat("Prices")
        self.P = pd_mat.shape[0]
        self.PRI_pg = {(p, g): pd_mat[p, g] for p in range(self.P) for g in range(self.G)}

        dr = pd.read_excel(xls, "Demand", header=None).values.flatten()
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
                    if val > max_val:
                        max_val = val
                    idx += 1
                self.M_rap[(r, a)] = max_val

        ud = get_mat("Upgrades")
        self.UPG_g1g2 = {}
        for g1 in range(self.G):
            for g2 in range(self.G):
                try:
                    self.UPG_g1g2[(g1, g2)] = int(ud[g1, g2])
                except Exception:
                    self.UPG_g1g2[(g1, g2)] = 1 if g1 == g2 else 0

        tc = get_mat("TransferCosts")
        tt = get_mat("TransferTimes")
        self.TC_gs1s2 = {}
        self.TT_s1s2 = {}
        for s1 in range(self.S):
            for s2 in range(self.S):
                self.TT_s1s2[(s1, s2)] = int(tt[s1, s2])
                for g in range(self.G):
                    # assumindo custos iguais para todos os grupos, como no original
                    self.TC_gs1s2[(g, s1, s2)] = float(tc[s1, s2])

        self.INX_gs = {(g, s): 0.0 for g in range(self.G) for s in range(self.S)}


# ==============================================================================
# 2. EVALUATOR (AGORA USANDO GUROBI "NORMAL", NÃO PERSISTENT)
# ==============================================================================
class PersistentEvaluator:  # mantive o nome para não mexer no resto do código
    def __init__(self, data, is_relaxed=False):
        type_str = "RELAXED (LP)" if is_relaxed else "INTEGER (MIP)"
        print(f"Building Model ({type_str})...")

        self.data = data
        self.model = ConcreteModel()
        m = self.model

        m.G = RangeSet(0, data.G - 1)
        m.S = RangeSet(0, data.S - 1)
        m.R = RangeSet(0, data.R - 1)
        m.A = RangeSet(0, data.A)
        m.P = RangeSet(0, data.P - 1)
        m.T = RangeSet(0, data.T)
        m.T_minus = RangeSet(0, data.T - 1)

        # domínio
        domain_type = NonNegativeReals if is_relaxed else NonNegativeIntegers

        # Capacity
        m.w_O = Var(m.G, m.S, domain=domain_type)
        m.w_L = Var(m.G, m.T_minus, m.S, domain=domain_type)

        # Flow (sempre relaxadas)
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

        data_obj = data

        def obj_rule(mod):
            rev = sum(
                mod.v[r, a, p] * data_obj.PRI_pg[p, data_obj.rental_gr[r]]
                for r in mod.R
                for a in mod.A
                for p in mod.P
            )
            cost_buy = sum(mod.w_O[g, s] * data_obj.COS_g[g] for g in mod.G for s in mod.S)
            cost_lease = sum(
                mod.f_L[g, t] * data_obj.LEA_g[g] for g in mod.G for t in mod.T_minus
            )
            cost_own = sum(
                mod.f_O[g, t] * data_obj.OWN_g[g] for g in mod.G for t in mod.T_minus
            )
            cost_trans = sum(
                (mod.y_L[s1, s2, g, t] + mod.y_O[s1, s2, g, t])
                * data_obj.TC_gs1s2[g, s1, s2]
                for s1 in mod.S
                for s2 in mod.S
                for g in mod.G
                for t in mod.T_minus
            )
            cost_upg = sum(
                (mod.u_L[r, a, g] + mod.u_O[r, a, g]) * data_obj.PYU
                for g in mod.G
                for r in mod.R
                for a in mod.A
                if data_obj.rental_gr[r] != g
            )
            return rev - cost_buy - cost_lease - cost_own - cost_trans - cost_upg

        m.obj = Objective(rule=obj_rule, sense=maximize)

        # Constraints
        m.c_U = Constraint(
            m.R,
            m.A,
            rule=lambda mod, r, a: mod.U[r, a]
            == sum(mod.v[r, a, p] for p in mod.P),
        )
        m.c1 = Constraint(
            m.R,
            m.A,
            rule=lambda mod, r, a: mod.U[r, a]
            == sum(mod.u_L[r, a, g] + mod.u_O[r, a, g] for g in mod.G),
        )

        def stock_O(mod, g, t, s):
            d = data_obj
            if t == 0:
                return mod.x_O[g, 0, s] == d.INX_gs[g, s] + mod.w_O[g, s]
            rin = sum(
                mod.u_O[r, a, g]
                for r in mod.R
                for a in mod.A
                if d.r_data[r, 1] == s + 1 and d.r_data[r, 3] == t
            )
            rout = sum(
                mod.u_O[r, a, g]
                for r in mod.R
                for a in mod.A
                if d.r_data[r, 0] == s + 1 and d.r_data[r, 2] == t
            )
            tin = sum(
                mod.y_O[s2, s, g, t - d.TT_s1s2[s2, s]]
                for s2 in mod.S
                if t - d.TT_s1s2[s2, s] in mod.T_minus
            )
            tout = sum(
                mod.y_O[s, s2, g, t] for s2 in mod.S if t in mod.T_minus
            )
            return (
                mod.x_O[g, t, s]
                == mod.x_O[g, t - 1, s] + rin - rout + tin - tout
            )

        m.c_stockO = Constraint(m.G, m.T, m.S, rule=stock_O)

        def stock_L(mod, g, t, s):
            d = data_obj
            if t == 0:
                return mod.x_L[g, 0, s] == 0
            rin = sum(
                mod.u_L[r, a, g]
                for r in mod.R
                for a in mod.A
                if d.r_data[r, 1] == s + 1 and d.r_data[r, 3] == t
            )
            rout = sum(
                mod.u_L[r, a, g]
                for r in mod.R
                for a in mod.A
                if d.r_data[r, 0] == s + 1 and d.r_data[r, 2] == t
            )
            tin = sum(
                mod.y_L[s2, s, g, t - d.TT_s1s2[s2, s]]
                for s2 in mod.S
                if t - d.TT_s1s2[s2, s] in mod.T_minus
            )
            tout = sum(
                mod.y_L[s, s2, g, t] for s2 in mod.S if t in mod.T_minus
            )
            acq = mod.w_L[g, t - 1, s] if (t - 1) in mod.T_minus else 0
            if t <= d.LP_g[g]:
                return (
                    mod.x_L[g, t, s]
                    == mod.x_L[g, t - 1, s] + acq + rin - rout + tin - tout
                )
            ret_idx = t - d.LP_g[g] - 1
            ret = mod.w_L[g, ret_idx, s] if ret_idx in mod.T_minus else 0
            return (
                mod.x_L[g, t, s]
                == mod.x_L[g, t - 1, s]
                + acq
                - ret
                + rin
                - rout
                + tin
                - tout
            )

        m.c_stockL = Constraint(m.G, m.T, m.S, rule=stock_L)

        m.c_cap = Constraint(
            m.G,
            m.T_minus,
            m.S,
            rule=lambda mod, g, t, s: sum(
                mod.u_L[r, a, g] + mod.u_O[r, a, g]
                for r in mod.R
                for a in mod.A
                if data_obj.r_data[r, 0] == s + 1 and data_obj.r_data[r, 2] == t
            )
            + sum(
                mod.y_L[s, s2, g, t] + mod.y_O[s, s2, g, t] for s2 in mod.S
            )
            <= mod.x_L[g, t, s] + mod.x_O[g, t, s],
        )

        m.c_upg = Constraint(
            m.R,
            m.A,
            m.G,
            rule=lambda mod, r, a, g: mod.u_L[r, a, g] + mod.u_O[r, a, g]
            == 0
            if data_obj.UPG_g1g2.get((data_obj.rental_gr[r], g), 0) == 0
            and data_obj.rental_gr[r] != g
            else Constraint.Skip,
        )

        m.c_fL = Constraint(
            m.G,
            m.T,
            rule=lambda mod, g, t: mod.f_L[g, t]
            >= sum(mod.x_L[g, t, s] for s in mod.S)
            + sum(
                mod.u_L[r, a, g]
                for r in mod.R
                for a in mod.A
                if data_obj.r_data[r, 2] <= t < data_obj.r_data[r, 3]
            ),
        )

        m.c_fO = Constraint(
            m.G,
            m.T,
            rule=lambda mod, g, t: mod.f_O[g, t]
            >= sum(mod.x_O[g, t, s] for s in mod.S)
            + sum(
                mod.u_O[r, a, g]
                for r in mod.R
                for a in mod.A
                if data_obj.r_data[r, 2] <= t < data_obj.r_data[r, 3]
            ),
        )

        m.c_bud = Constraint(
            rule=lambda mod: sum(
                mod.w_O[g, s] * data_obj.COS_g[g] for g in mod.G for s in mod.S
            )
            <= data_obj.BUD
        )

        print("  Model built.")

        # Solver NORMAL, não persistent
        self.opt = SolverFactory("gurobi")
        self.opt.options["OutputFlag"] = 0
        self.opt.options["MIPGap"] = 0.01
        self.is_relaxed = is_relaxed

    def solve_for_pricing(self, pricing_policy):
        """
        Ajusta os upper bounds de v[r,a,p] conforme pricing_policy
        e resolve a MIP/LP usando o Gurobi padrão.
        """
        m = self.model
        data = self.data

        for r in m.R:
            for a in m.A:
                target = pricing_policy.get((r, a), -1)
                for p in m.P:
                    ub = data.DEM_rap[(r, a, p)] if p == target else 0.0
                    m.v[r, a, p].setub(ub)

        res = self.opt.solve(m, tee=False)

        tc = res.solver.termination_condition
        if tc != TerminationCondition.optimal:
            print(f"  Solve not optimal (tc={tc}). Returning -1e9.")
            return -1e9, {}

        # Agora todas variáveis têm value
        obj_val = value(m.obj)

        current_owned = {}
        # Só extrai frota se modelo for inteiro:
        if not self.is_relaxed:
            for g in m.G:
                for s in m.S:
                    if value(m.w_O[g, s]) > 0.5:
                        current_owned[(g, s)] = int(round(value(m.w_O[g, s])))

        return obj_val, current_owned


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
                rev = data.PRI_pg[p, data.rental_gr[r]] * data.DEM_rap[(r, a, p)]
                if rev > best_rev:
                    best_rev = rev
                    best_p = p
            target_p[(r, a)] = best_p
    return target_p


def perturb_pricing(pricing_in, P_max, mutation_rate):
    pricing = copy.deepcopy(pricing_in)
    keys = list(pricing.keys())
    num_changes = max(1, int(len(keys) * mutation_rate))
    targets = random.sample(keys, num_changes)
    for key in targets:
        pricing[key] = random.randint(0, P_max - 1)
    return pricing


def save_result(filename, w_owned_dict, pricing_dict):
    print(f"\nSaving final solution to {filename}...")
    o_data = []
    p_data = []
    for (g, s), val in w_owned_dict.items():
        o_data.append({"Group": g + 1, "Station": s + 1, "Quantity": val})
    for (r, a), p in pricing_dict.items():
        p_data.append({"RentalID": r, "Antecedence": a, "PriceLevel": p})
    with pd.ExcelWriter(filename) as writer:
        pd.DataFrame(o_data).to_excel(
            writer, sheet_name="Owned_Capacity", index=False
        )
        pd.DataFrame(p_data).to_excel(
            writer, sheet_name="Pricing_Policy", index=False
        )


# ==============================================================================
# 4. SA MAIN LOOP
# ==============================================================================
def run_sa_fast_relaxation(instance, max_seconds=120):
    print(f"Running SA (Reset @ 10 Iters + Saturation Drop) for {instance}")

    data = InstanceData(instance)

    evaluator_lp = PersistentEvaluator(data, is_relaxed=True)
    evaluator_mip = PersistentEvaluator(data, is_relaxed=False)

    # 2. Solução inicial: Excel
    PRICE_EXCEL_FILE = "heuristic_price_from_capacity_all_instances.xlsx"

    try:
        print("\nCarregando preços iniciais do Excel...")

        instance_sheet = os.path.basename(instance).replace(".xlsx", "")
        xls = pd.ExcelFile(PRICE_EXCEL_FILE)

        if instance_sheet not in xls.sheet_names:
            print(
                f"A aba '{instance_sheet}' não existe — procurando correspondência..."
            )
            match = None
            inst_num = "".join(ch for ch in instance_sheet if ch.isdigit())
            for sheet in xls.sheet_names:
                if sheet.startswith("Inst") and inst_num in sheet:
                    match = sheet
                    break
            if match:
                print(f"> Usando aba encontrada: {match}")
                instance_sheet = match
            else:
                raise ValueError(
                    f"Nenhuma aba correspondente encontrada para '{instance_sheet}'."
                )

        df_price = load_constructive_price_from_excel(
            PRICE_EXCEL_FILE, instance_sheet
        )

        curr_price = {}
        for _, row in df_price.iterrows():
            r = int(row["rental_type"]) - 1
            a = int(row["antecedence"])
            p = int(row["p_init"]) - 1
            if p < 0:
                p = 0
            if p >= data.P:
                p = data.P - 1
            curr_price[(r, a)] = p

        if len(curr_price) < (data.R * (data.A + 1)):
            greedy = generate_greedy_pricing(data)
            for key, val in greedy.items():
                if key not in curr_price:
                    curr_price[key] = val

        print("Inicialização via Excel concluída.")

    except Exception as e:
        print(f"Falha ao ler Excel ({e}). Usando greedy.")
        curr_price = generate_greedy_pricing(data)

    curr_obj, _ = evaluator_lp.solve_for_pricing(curr_price)
    best_obj_lp = curr_obj
    best_price = copy.deepcopy(curr_price)

    MAX_MUT_RATE = 0.50
    MIN_MUT_RATE = 0.05
    STAGNATION_LIMIT = 10
    SATURATION_POINT = 3.0

    current_mutation_rate = MAX_MUT_RATE
    amplitude = MAX_MUT_RATE - MIN_MUT_RATE

    T = 50000
    alpha = 0.95
    start_time = time.time()
    total_iter = 0
    iter_since_best = 0

    print(
        f"\n{'Iter':<5} | {'Temp':<8} | {'Mut%':<6} | {'New Profit':<14} | {'Best Profit':<14} | {'Note'}"
    )
    print("-" * 100)

    while (time.time() - start_time) < max_seconds:
        total_iter += 1
        iter_since_best += 1
        note_str = "-"

        if iter_since_best >= STAGNATION_LIMIT:
            current_mutation_rate = MAX_MUT_RATE
            iter_since_best = 0
            note_str = "RESET (Stuck)"

        n_price = perturb_pricing(
            curr_price, data.P, mutation_rate=current_mutation_rate
        )
        n_obj, _ = evaluator_lp.solve_for_pricing(n_price)

        delta = n_obj - curr_obj
        accepted = False
        if delta > 0:
            accepted = True
        elif n_obj > -1e8:
            try:
                prob = math.exp(delta / T)
            except Exception:
                prob = 0
            if random.random() < prob:
                accepted = True

        if accepted:
            curr_price, curr_obj = n_price, n_obj
            if curr_obj > best_obj_lp:
                iter_since_best = 0
                if best_obj_lp == 0:
                    improvement_pct = 1.0
                else:
                    improvement_pct = (curr_obj - best_obj_lp) / abs(best_obj_lp)
                best_obj_lp = curr_obj
                best_price = copy.deepcopy(curr_price)
                note_str = f"+{improvement_pct*100:.2f}%"
                ratio = improvement_pct / SATURATION_POINT
                target_rate = MIN_MUT_RATE + (ratio * amplitude)
                current_mutation_rate = (current_mutation_rate + target_rate) / 2
                current_mutation_rate = max(
                    MIN_MUT_RATE, min(MAX_MUT_RATE, current_mutation_rate)
                )

        mut_str = f"{current_mutation_rate*100:.1f}%"
        print(
            f"{total_iter:<5} | {int(T):<8} | {mut_str:<6} | {n_obj:<14,.0f} | {best_obj_lp:<14,.0f} | {note_str}"
        )

        T *= alpha

    print(f"\nSA Loop Finished. Best Relaxed Profit: {best_obj_lp:,.0f}")

    print("\n--- FINAL VALIDATION (MIP) ---")
    final_obj_mip, final_fleet = evaluator_mip.solve_for_pricing(best_price)
    print(f"Final Real (Integer) Profit: {final_obj_mip:,.0f}")

    save_result("BestSolution_SaturationSA.xlsx", final_fleet, best_price)


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    INSTANCE = r"Inst40.xlsx"
    if os.path.exists(INSTANCE):
        run_sa_fast_relaxation(INSTANCE, max_seconds=1200)
    else:
        print("Instance not found.")
# ==============================================================================