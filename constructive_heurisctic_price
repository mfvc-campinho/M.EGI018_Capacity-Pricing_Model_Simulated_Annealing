import os
import math
import pandas as pd
import numpy as np
import sys

print("CWD =", os.getcwd())
print(sys.executable)

# IMPORTA A CONSTRUCTIVE DE CAPACIDADE
from constructive_heuristic import (
    load_unit_params,
    load_parameters_by_group,
    load_rental_types,
    load_demands,            # não é usada aqui, mas mantida se quiser comparar
    compute_D_matrix,        # idem
    heuristic_capacity_instance,
)

# ======================================================
# 0. Leitura COMPLETA da matriz DEM_{r,a,p} da aba Demand
# ======================================================

def load_full_demand_tensor(xls: pd.ExcelFile, unit: dict):
    """
    Lê a aba 'Demand' e reconstrói a matriz DEM[r,a,p].

    Formato assumido (compatível com o dataset que você descreveu):
      - Primeira linha: header, com colunas:
          DEM_rap, (p=1,a=0), (p=1,a=1), ..., (p=1,a=A),
                    (p=2,a=0), ..., (p=2,a=A),
                    ...
                    (p=P,a=A)
      - Primeira coluna (índice 0): identificador do rental type (ignoramos conteúdo)
      - Demais colunas: valores de DEM_{r,a,p}.

    Parâmetros:
      xls  : pd.ExcelFile já aberto
      unit : dict vindo de load_unit_params (contém A e, opcionalmente, P)

    Retorna:
      DEM: numpy array de shape (R, A+1, P+1), 0-based em r,a,p,
           mas com p iniciando em 1 na última dimensão:
              DEM[r, a, p] válido para:
                r = 0..R-1,
                a = 0..A,
                p = 1..P.
    """

    A = int(unit["A"])

    dem_raw = pd.read_excel(xls, "Demand", header=0)
    # Ignora primeira coluna (DEM_rap ou ID do rental type)
    body = dem_raw.iloc[:, 1:].to_numpy(dtype=float)  # shape (R, num_cols)

    R, num_cols = body.shape

    # Se P não estiver em UnitParameters, inferimos a partir do nº de colunas
    if "P" in unit:
        P = int(unit["P"])
        expected_cols = P * (A + 1)
        if expected_cols != num_cols:
            raise ValueError(
                f"Inconsistent Demand columns: expected {expected_cols}, "
                f"found {num_cols}. Check A and P in UnitParameters."
            )
    else:
        # Inferir P pela quantidade de colunas
        if num_cols % (A + 1) != 0:
            raise ValueError(
                "Cannot infer P from Demand: num_cols is not multiple of (A+1)."
            )
        P = num_cols // (A + 1)

    # DEM[r,a,p], com p em 1..P (p=0 fica sempre 0, por conveniência)
    DEM = np.zeros((R, A + 1, P + 1), dtype=float)

    for r in range(R):
        for j in range(num_cols):
            val = body[r, j]
            p = j // (A + 1) + 1     # p = 1..P
            a = j % (A + 1)          # a = 0..A
            DEM[r, a, p] = val

    return DEM, P, R


# =========================================
# 1. Constructive Heuristic de PREÇO (detalhada)
# =========================================

def heuristic_price_from_cap_heuristic(
    path,
    tau_low=0.5,
    tau_high=0.9,
    gamma_cap=0.7,
    penalty_weight=1.0,
):
    """
    Constructive heuristic for PRICE using:

    - Full demand tensor DEM_{r,a,p} from the 'Demand' sheet,
    - Capacity CAP[g,s,t] from the constructive capacity heuristic,
    - A utilization band [tau_low, tau_high] as 'desirable' usage level.

    For each rental type r and antecedence a (a = 0..A), this heuristic:
      1. Evaluates all price levels p = 1..P,
      2. Computes an approximate utilization and revenue proxy,
      3. Penalizes under- and over-utilization relative to [tau_low, tau_high],
      4. Selects the p that maximizes a score function.

    Output:
      DataFrame with columns:
        - rental_type  (1-based index)
        - antecedence  (a = 0..A, with a=0 = walk-in)
        - p_init       (chosen initial price level in {1,...,P})
    """

    xls = pd.ExcelFile(path)

    # 1) Carrega parâmetros e dados básicos
    unit = load_unit_params(xls)
    G, LEA, LP, COS, OWN = load_parameters_by_group(xls)
    rt   = load_rental_types(xls)

    S = int(unit["S"])
    T = int(unit["T"])
    A = int(unit["A"])

    # 2) Carrega DEM_{r,a,p} completa
    DEM, P, R = load_full_demand_tensor(xls, unit)

    # 3) Obtem capacidade CAP[g,s,t] da constructive de capacidade
    (
        _unit_c,
        Gc,
        Sc,
        Tc,
        Freq,
        w_owned,
        w_leased,
        PYU,
        phi,
    ) = heuristic_capacity_instance(path, gamma=gamma_cap)

    assert G == Gc and S == Sc and T == Tc, "Dimensões inconsistentes entre dados e capacidade."

    # 4) Reconstrói leasing ativo ao longo do horizonte
    ActiveLease = np.zeros((G + 1, S + 1, T), dtype=int)
    for g in range(1, G + 1):
        Lg = int(LP[g - 1])  # duração de leasing para grupo g
        for s in range(1, S + 1):
            for t in range(T):
                q = int(w_leased[g, t, s])
                if q > 0:
                    for tau in range(t, min(T, t + Lg)):
                        ActiveLease[g, s, tau] += q

    # 5) CAP[g,s,t] = frota própria + leasing ativo
    CAP = np.zeros((G + 1, S + 1, T), dtype=float)
    for g in range(1, G + 1):
        for s in range(1, S + 1):
            for t in range(T):
                CAP[g, s, t] = int(w_owned[g, s]) + ActiveLease[g, s, t]

    # 6) Para cada rental type r e antecedence a, escolher p que maximiza score
    price_records = []
    eps = 1e-6

    # Penalty band: queremos utilização entre tau_low e tau_high
    tau_low = float(tau_low)
    tau_high = float(tau_high)

    for idx, row in rt.iterrows():
        g_r = int(row["gr"])
        s_r = int(row["sout"])
        dout = int(row["dout"])
        din  = int(row["din"])

        # Janela de uso no horizonte
        first_t = max(dout, 0)
        last_t  = min(din - 1, T - 1)
        if last_t < first_t:
            # Se a janela é inválida, não consome capacidade; usamos preço mínimo ou máximo?
            # Aqui, assumimos p_init = 1 (irrelevante na prática se não há uso).
            for a in range(A + 1):
                price_records.append({
                    "rental_type": idx + 1,
                    "antecedence": a,
                    "p_init": 1,
                })
            continue

        # Pré-calcula capacidade mínima na janela (independe de p, mas depende do grupo/estação)
        cap_window = [CAP[g_r, s_r, t] for t in range(first_t, last_t + 1)]
        cap_min = float(min(cap_window)) if len(cap_window) > 0 else 0.0

        for a in range(A + 1):
            best_p = 1
            best_score = -1e18  # bem negativo

            # Caso extremo: se cap_min ~ 0 para toda a janela, usar preço alto para racionar
            if cap_min <= eps:
                price_records.append({
                    "rental_type": idx + 1,
                    "antecedence": a,
                    "p_init": P,
                })
                continue

            for p in range(1, P + 1):
                dem_rap = DEM[idx, a, p]

                # Se não há demanda nesse (r,a,p), score tende a ser ruim,
                # mas ainda podemos definir um preço (provavelmente o mais alto).
                if dem_rap <= eps:
                    # Score muito baixo, mas finito
                    # (se todos forem assim, acabaremos com best_p = P no fim)
                    score = -1e6
                else:
                    # Quantidade efetivamente atendível limitada pela capacidade mínima
                    q_eff = min(dem_rap, cap_min)
                    # Utilização aproximada nessa janela (0..1 ou >1 se extrapolar)
                    utilization = q_eff / (cap_min + eps)

                    # Proxy de receita: assumimos receita ∝ nível de preço * quantidade atendida
                    revenue_proxy = p * q_eff

                    # Penalização de utilização fora da banda [tau_low, tau_high]
                    penalty = 0.0
                    if utilization < tau_low:
                        penalty += (tau_low - utilization) * cap_min
                    if utilization > tau_high:
                        penalty += (utilization - tau_high) * cap_min

                    score = revenue_proxy - penalty_weight * penalty

                # Atualiza melhor preço
                if score > best_score:
                    best_score = score
                    best_p = p

            # Garantia de que o nível está em [1,P]
            best_p = max(1, min(best_p, P))

            price_records.append({
                "rental_type": idx + 1,  # 1-based
                "antecedence": a,        # a = 0..A (0 = walk-in)
                "p_init": best_p,
            })

    df_price = pd.DataFrame(price_records,
                            columns=["rental_type", "antecedence", "p_init"])
    return df_price


# =========================================
# 2. Rodar para todas as instâncias
# =========================================

def run_price_heuristic_for_all_instances(
    folder_path,
    output_excel,
    tau_low=0.5,
    tau_high=0.9,
    gamma_cap=0.7,
    penalty_weight=1.0,
):
    """
    Aplica a heuristic_price_from_cap_heuristic a TODAS as instâncias InstX.xlsx
    da pasta folder_path e grava um único arquivo Excel com:
      - Uma aba por instância, contendo (r,a,p_init),
      - Uma aba 'summary' com um resumo simples.
    """

    paths = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".xlsx") and f.startswith("Inst")
        ],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("Inst", ""))
    )

    summary = []
    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        for p in paths:
            name = os.path.basename(p).replace(".xlsx", "")
            print("Processing price heuristic for", name)

            df = heuristic_price_from_cap_heuristic(
                p,
                tau_low=tau_low,
                tau_high=tau_high,
                gamma_cap=gamma_cap,
                penalty_weight=penalty_weight,
            )

            df.to_excel(writer, sheet_name=name, index=False)

            summary.append({
                "instance": name,
                "num_rental_types": df["rental_type"].nunique(),
                "num_antecedence_levels": df["antecedence"].nunique(),
            })

        pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)


if __name__ == "__main__":
    folder = "data"
    output_excel = "heuristic_price_from_capacity_all_instances.xlsx"

    run_price_heuristic_for_all_instances(
        folder,
        output_excel,
        tau_low=0.5,        # banda de utilização desejada (pode calibrar)
        tau_high=0.9,
        gamma_cap=0.7,      # mesmo gamma da constructive de capacidade
        penalty_weight=1.0, # peso da penalização de (sub/sobre)utilização
    )

    print("Initial price file generated:", output_excel)
