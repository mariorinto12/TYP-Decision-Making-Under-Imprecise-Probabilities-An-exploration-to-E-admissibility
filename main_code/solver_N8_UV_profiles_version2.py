import numpy as np
from scipy.optimize import linprog


# ============================================================
# BLOCK 1 — Generic linear constraint container (solver form)
# ============================================================

class ConstraintSet:
    """
    Stores constraints in scipy.linprog form:

      A_ub x <= b_ub
      A_eq x == b_eq
      bounds: (low, high) per variable

    This is just an internal representation layer so the LP-building
    steps can read like the mathematical formulation.
    """

    def __init__(self, number_of_variables: int):
        self.number_of_variables = int(number_of_variables)

        self.A_ub = []
        self.b_ub = []

        self.A_eq = []
        self.b_eq = []

        self.bounds = [(None, None) for _ in range(self.number_of_variables)]

    def add_upper_bound_constraint(self, coefficient_row, bound_value) -> None:
        row = np.asarray(coefficient_row, dtype=float).reshape(-1)
        if row.size != self.number_of_variables:
            raise ValueError("Row length must equal number_of_variables.")
        self.A_ub.append(row)
        self.b_ub.append(float(bound_value))

    def add_equality_constraint(self, coefficient_row, bound_value) -> None:
        row = np.asarray(coefficient_row, dtype=float).reshape(-1)
        if row.size != self.number_of_variables:
            raise ValueError("Row length must equal number_of_variables.")
        self.A_eq.append(row)
        self.b_eq.append(float(bound_value))

    def set_variable_bounds(self, variable_index: int, lower_bound, upper_bound) -> None:
        if variable_index < 0 or variable_index >= self.number_of_variables:
            raise IndexError("Variable index out of range.")
        self.bounds[variable_index] = (lower_bound, upper_bound)

    def to_solver_matrices(self):
        A_ub = np.vstack(self.A_ub) if self.A_ub else None
        b_ub = np.asarray(self.b_ub, dtype=float) if self.b_ub else None

        A_eq = np.vstack(self.A_eq) if self.A_eq else None
        b_eq = np.asarray(self.b_eq, dtype=float) if self.b_eq else None

        return A_ub, b_ub, A_eq, b_eq, self.bounds


# ============================================================
# BLOCK 2 — LP solver (maximisation wrapper over linprog)
# ============================================================

def solve_linear_program_maximisation(
    objective_coefficients,
    constraint_set: ConstraintSet,
    tolerance: float = 1e-9
) -> dict:
    """
    Maximise c^T x subject to the constraints in constraint_set.

    scipy.linprog solves minimisation, so we pass -c and negate the optimum.
    """
    c = np.asarray(objective_coefficients, dtype=float).reshape(-1)
    if c.size != constraint_set.number_of_variables:
        raise ValueError("Objective length must match number_of_variables.")

    A_ub, b_ub, A_eq, b_eq, bounds = constraint_set.to_solver_matrices()

    res = linprog(
        c=-c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    x = None
    if res.x is not None:
        x = np.asarray(res.x, dtype=float)
        x[np.abs(x) < tolerance] = 0.0

    return {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "solution": x,
        "optimal_value": None if res.fun is None else float(-res.fun),
    }


# ============================================================
# BLOCK 3 — Small LP building blocks (dot-product constraints)
# ============================================================

def add_non_negativity_constraint(constraint_set: ConstraintSet, coefficient_row) -> None:
    """
    Add:  coefficient_row · x >= 0
    as:  (-coefficient_row) · x <= 0
    """
    a = np.asarray(coefficient_row, dtype=float).reshape(-1)
    constraint_set.add_upper_bound_constraint((-a).tolist(), 0.0)


def add_interval_constraint(
    constraint_set: ConstraintSet,
    coefficient_row,
    lower_bound: float,
    upper_bound: float
) -> None:
    """
    Add: lower_bound <= coefficient_row · x <= upper_bound
    """
    a = np.asarray(coefficient_row, dtype=float).reshape(-1)
    constraint_set.add_upper_bound_constraint(a.tolist(), float(upper_bound))
    constraint_set.add_upper_bound_constraint((-a).tolist(), float(-lower_bound))


# ============================================================
# BLOCK 4 — Baseline domain for beliefs: relaxed simplex
# ============================================================

def build_relaxed_simplex(number_of_states: int) -> ConstraintSet:
    """
    Relaxed simplex for pi over states:

      pi_j >= 0
      sum_j pi_j <= 1

    The objective max sum_j pi_j is used later to detect whether we can reach 1
    (i.e., a proper distribution exists inside the feasible region).
    """
    cs = ConstraintSet(number_of_states)

    cs.add_upper_bound_constraint([1.0] * number_of_states, 1.0)  # sum pi <= 1

    for j in range(number_of_states):
        cs.set_variable_bounds(j, 0.0, None)  # pi_j >= 0

    return cs


# ============================================================
# BLOCK 5 — Credal set M: expectation bounds + optional pi cap
# ============================================================

def add_credal_set_constraints(constraint_set: ConstraintSet, credal_set: dict) -> None:
    """
    Credal set constraints of the form:

      b_low[r] <= F[r] · pi <= b_up[r],   r = 1..R

    plus (optionally) component-wise cap:

      0 <= pi_j <= pi_cap,   j = 1..m

    If you include pi_cap in the credal_set dict, it becomes part of M by design.
    """
    F = np.asarray(credal_set["F"], dtype=float)
    b_low = np.asarray(credal_set["b_low"], dtype=float).reshape(-1)
    b_up = np.asarray(credal_set["b_up"], dtype=float).reshape(-1)

    if F.ndim == 1:
        F = F.reshape(1, -1)

    if F.shape[1] != constraint_set.number_of_variables:
        raise ValueError("Credal F must have #columns equal to #states (n_vars).")

    if b_low.size != F.shape[0] or b_up.size != F.shape[0]:
        raise ValueError("Credal b_low and b_up must match #rows of F.")

    for r in range(F.shape[0]):
        add_interval_constraint(
            constraint_set=constraint_set,
            coefficient_row=F[r],
            lower_bound=b_low[r],
            upper_bound=b_up[r],
        )

    pi_cap = credal_set.get("pi_cap", None)
    if pi_cap is not None:
        cap = float(pi_cap)
        for j in range(constraint_set.number_of_variables):
            constraint_set.set_variable_bounds(j, 0.0, cap)


# ============================================================
# BLOCK 6 — E-admissibility region constraints (optimality)
# ============================================================

def add_e_admissibility_constraints(constraint_set: ConstraintSet, utility_matrix, action_index: int) -> None:
    """
    E-admissibility constraints for action i:

      (U[i] - U[k]) · pi >= 0   for all k != i

    This enforces that action i has expected utility at least as large as every competitor,
    for the same belief pi.
    """
    U = np.asarray(utility_matrix, dtype=float)
    n_actions, n_states = U.shape

    if constraint_set.number_of_variables != n_states:
        raise ValueError("ConstraintSet dimension must match number of states.")

    if action_index < 0 or action_index >= n_actions:
        raise IndexError("Action index out of range.")

    Ui = U[action_index]

    for k in range(n_actions):
        if k == action_index:
            continue
        add_non_negativity_constraint(constraint_set, Ui - U[k])


# ============================================================
# BLOCK 7 — LP builder for E-admissibility test
# ============================================================

def build_e_admissibility_linear_program(utility_matrix, action_index: int, credal_set=None) -> ConstraintSet:
    """
    Build the feasible region for the LP test:

      pi in relaxed simplex
      pi in M (if credal_set is provided)
      (U[i]-U[k])·pi >= 0 for all k != i

    Then we solve: max sum_j pi_j
    and declare E-admissible if optimum reaches 1 (within tolerance).
    """
    U = np.asarray(utility_matrix, dtype=float)
    _, m = U.shape

    cs = build_relaxed_simplex(m)

    if credal_set is not None:
        add_credal_set_constraints(cs, credal_set)

    add_e_admissibility_constraints(cs, U, action_index)

    return cs


# ============================================================
# BLOCK 8 — E-admissibility check (solve and decide)
# ============================================================

def is_e_admissible(utility_matrix, action_index: int, credal_set=None, tolerance: float = 1e-9):
    """
    Solve:

      max 1^T pi
      s.t. pi feasible (constraints built above)

    Interpretation:
      - If optimum >= 1 - tol: there exists a feasible pi with sum(pi)=1,
        hence action is E-admissible (under the chosen domain / credal set).
    """
    cs = build_e_admissibility_linear_program(utility_matrix, action_index, credal_set=credal_set)

    res = solve_linear_program_maximisation(
        objective_coefficients=[1.0] * cs.number_of_variables,
        constraint_set=cs,
        tolerance=tolerance,
    )

    opt = res["optimal_value"]
    admissible = (opt is not None) and (opt >= 1.0 - tolerance)

    return admissible, res


# ============================================================
# BLOCK 9 — Enumerate all E-admissible actions
# ============================================================

def all_e_admissible_actions(utility_matrix, credal_set=None, tolerance: float = 1e-9, return_witnesses: bool = True):
    U = np.asarray(utility_matrix, dtype=float)
    n_actions, _ = U.shape

    admissible_actions = []
    witnesses = {}

    for i in range(n_actions):
        admissible, res = is_e_admissible(U, i, credal_set=credal_set, tolerance=tolerance)
        if admissible:
            admissible_actions.append(i)
            if return_witnesses:
                witnesses[i] = res["solution"]

    return (admissible_actions, witnesses) if return_witnesses else admissible_actions


# ============================================================
# BLOCK 10 — Example data (Simulation 5)
# ============================================================

def get_simulation_5_problem_data():
    portfolio_names = [
        "Balanced",
        "Conservative",
        "Aggressive",
        "MarketIndex",
        "GoldHedge",
        "BondHeavy",
        "TechTravel",
        "AntiCrisis",
    ]

    state_names = ["S1", "S2", "S3", "S4", "S5"]

    utility_matrix = np.array([
        [40, 25, 20, 39, 27], # Balanced
        [38, 37, 31, 22, 21], # Conservative
        [20, 37, 31, 41, 16], # Aggressive
        [39, 35, 39, 39, 34], # MarketIndex
        [33, 31, 30, 31, 31], # GoldHedge
        [11, 10, 13, 13, 12], # BondHeavy
        [23, 54, 16, 32, 45], # TechTravel
        [23, 21, 24, 23, 25], # AntiCrisis
    ], dtype=float)

    F = np.array([
        [28.3750, 31.2500, 25.5000, 30.0000, 26.3750],  # f4 - Mean Utility
        [0.7071,  0.7806,  0.4330,  0.5995,  0.5995],   # f7 - Ordinal Standard Deviation
        [29.0000, 44.0000, 26.0000, 28.0000, 33.0000],  # f8 - Range of Utilities
    ], dtype=float)

    return portfolio_names, state_names, utility_matrix, F


def get_profiles(F):
    credal_0 = {
        "F": F,
        "b_low": [26.0, 0.45, 27.0],
        "b_up":  [30.0, 0.65, 35.0],
        "pi_cap": 0.45,
    }

    credal_A = {
        "F": F,
        "b_low": [27.0, 0.43, 26.0],
        "b_up":  [30.0, 0.58, 30.0],
        "pi_cap": 0.35,
    }

    credal_B = {
        "F": F,
        "b_low": [27.5, 0.52, 28.0],
        "b_up":  [31.25, 0.70, 36.0],
        "pi_cap": 0.45,
    }

    credal_C = {
        "F": F,
        "b_low": [29.0, 0.60, 33.0],
        "b_up":  [31.25, 0.80, 44.0],
        "pi_cap": None,
    }

    profile_labels = {
        "credal_0": "Baseline",
        "A": "Decisor A: Capital Defence (High Ambiguity Aversion)",
        "B": "Decisor B: Core Allocation (Moderate Ambiguity Aversion)",
        "C": "Decisor C: Seeking Opportunity (Low Ambiguity Aversion)",
    }

    profiles = {
        "credal_0": credal_0,
        "A": credal_A,
        "B": credal_B,
        "C": credal_C,
    }

    return profiles, profile_labels


# ============================================================
# BLOCK 11 — Printing helpers (your exact output format)
# ============================================================

def print_eadmissibility_results(portfolio_names, admissible_indices, witnesses, state_names):
    print("\n=== E-ADMISSIBILITY RESULTS ===\n")

    if len(admissible_indices) == 0:
        print("No E-admissible portfolios found.")
        return

    print("E-admissible portfolios:")
    for idx in admissible_indices:
        print(f"  - {idx}: {portfolio_names[idx]}")
    print("")

    print("Witness distributions (pi*) for each E-admissible portfolio:")
    for idx in admissible_indices:
        x = witnesses.get(idx, None)
        if x is None:
            continue

        formatted = ", ".join([f"{state_names[j]}={x[j]:.4f}" for j in range(len(x))])
        print(f"  - {portfolio_names[idx]}: {formatted}   (sum={x.sum():.4f})")


def print_full_eadmissibility_diagnostics(
    portfolio_names,
    state_names,
    utility_matrix,
    credal_set=None,
    tolerance: float = 1e-9,
    print_near_miss: bool = True,
    min_opt_to_print: float = 0.8
):
    header = "WITH credal set" if credal_set is not None else "NO credal set"
    print(f"\n=== FULL E-ADMISSIBILITY DIAGNOSTICS ({header}) ===\n")

    U = np.asarray(utility_matrix, dtype=float)
    n_actions = U.shape[0]

    print(f"{'Idx':>3} | {'Portfolio':<18} | {'Opt':>9} | {'Sum(pi*)':>8} | {'OK':>2} | Status")
    print("-" * 72)

    for i in range(n_actions):
        admissible, res = is_e_admissible(U, i, credal_set=credal_set, tolerance=tolerance)

        opt = res["optimal_value"]
        solver_ok = res["success"]

        x = res["solution"]
        sum_pi = float(x.sum()) if x is not None else float("nan")

        opt_str = "infeasible" if not solver_ok else f"{opt:>10.4f}"
        status_label = "E-adm" if admissible else "No"

        print(
            f"{i:>3} | "
            f"{portfolio_names[i]:<18} | "
            f"{opt_str:>9} | "
            f"{sum_pi:>8.4f} | "
            f"{'Y' if solver_ok else 'N':>2} | "
            f"{status_label}"
        )
        
        # --- Near-miss witness (feasible but not E-admissible because sum(pi*) < 1) ---
        if print_near_miss and solver_ok and (x is not None) and (opt is not None):
            if (opt > min_opt_to_print) and (opt < 1.0 - tolerance):
                formatted = ", ".join([f"{state_names[j]}={x[j]:.4f}" for j in range(len(x))])
                print(f"      near-miss pi*: {formatted}   (sum={x.sum():.4f})")


# ============================================================
# BLOCK 12 - Construction of diverse decisor profiles
# ============================================================

def run_profile(
    profile_name: str,
    portfolio_names,
    state_names,
    utility_matrix,
    credal_set,
    tol: float = 1e-9
):
    print("\n" + "=" * 78)
    print(f"PROFILE: {profile_name}")
    print("=" * 78)

    # Diagnostics WITH this credal set
    print_full_eadmissibility_diagnostics(
        portfolio_names=portfolio_names,
        state_names=state_names,
        utility_matrix=utility_matrix,
        credal_set=credal_set,
        tolerance=tol,
    )
    
    admissible, witnesses = all_e_admissible_actions(
        utility_matrix=utility_matrix,
        credal_set=credal_set,
        tolerance=tol,
        return_witnesses=True,
    )

    print_eadmissibility_results(
        portfolio_names=portfolio_names,
        admissible_indices=admissible,
        witnesses=witnesses,
        state_names=state_names,
    )

    # Return names for summary table
    admissible_names = [portfolio_names[i] for i in admissible]
    return admissible_names


def print_summary_matrix(profile_to_admissibles: dict, all_portfolio_names: list):
    print("\n" + "=" * 78)
    print("SUMMARY (E-admissible by profile)")
    print("=" * 78)

    profiles = list(profile_to_admissibles.keys())
    col_w = max(12, max(len(p) for p in profiles) + 2)

    header = "Portfolio".ljust(20) + "".join([p.ljust(col_w) for p in profiles])
    print(header)
    print("-" * len(header))

    for name in all_portfolio_names:
        row = name.ljust(20)
        for p in profiles:
            mark = "YES" if name in profile_to_admissibles[p] else "-"
            row += mark.ljust(col_w)
        print(row)


# ============================================================
# BLOCK X — Export witnesses / near-miss for the visualizer
# ============================================================

def get_profile_witnesses(
    profile_key: str,
    tolerance: float = 1e-9,
    min_opt_to_store: float = 0.8
) -> dict:
    """
    Returns data for the visualizer.

    Output format:
      {
        "witnesses": { action_idx: full_pi (size m) },
        "near_miss": { action_idx: full_pi (size m) },
      }

    - "witnesses"  -> E-admissible actions (opt >= 1 - tol) with pi* summing ~1
    - "near_miss"  -> feasible but not E-admissible because opt in (min_opt, 1-tol)
    """
    portfolio_names, state_names, utility_matrix, F = get_simulation_5_problem_data()
    profiles, _ = get_profiles(F)

    if profile_key not in profiles:
        raise KeyError(f"Unknown profile_key: {profile_key}")

    credal_set = profiles[profile_key]

    U = np.asarray(utility_matrix, dtype=float)
    n_actions = U.shape[0]

    witnesses = {}
    near_miss = {}

    for i in range(n_actions):
        admissible, res = is_e_admissible(
            utility_matrix=U,
            action_index=i,
            credal_set=credal_set,
            tolerance=tolerance
        )

        solver_ok = bool(res.get("success", False))
        x = res.get("solution", None)
        opt = res.get("optimal_value", None)

        if (not solver_ok) or (x is None) or (opt is None):
            continue

        # E-admissible witness (tau ~ 1)
        if opt >= 1.0 - tolerance:
            witnesses[i] = np.asarray(x, dtype=float)
            continue

        # Near-miss witness (tau < 1 but "closest" feasible)
        if (opt > min_opt_to_store) and (opt < 1.0 - tolerance):
            near_miss[i] = np.asarray(x, dtype=float)

    return {
        "witnesses": witnesses,
        "near_miss": near_miss,
    }

# ============================================================
# BLOCK 13 — Main (run: no credal set, then with credal set)
# ============================================================

def main():
    portfolio_names, state_names, utility_matrix, F = get_simulation_5_problem_data()
    profiles, profile_labels = get_profiles(F)

    tol = 1e-9

    # ============================================================
    # 0) Baseline: NO credal set
    # ============================================================

    print_full_eadmissibility_diagnostics(
        portfolio_names=portfolio_names,
        state_names=state_names,
        utility_matrix=utility_matrix,
        credal_set=None,
        tolerance=tol,
    )

    admissible_no_credal, witnesses_no_credal = all_e_admissible_actions(
        utility_matrix=utility_matrix,
        credal_set=None,
        tolerance=tol,
        return_witnesses=True,
    )

    print_eadmissibility_results(
        portfolio_names=portfolio_names,
        admissible_indices=admissible_no_credal,
        witnesses=witnesses_no_credal,
        state_names=state_names,
    )

    profile_to_admissibles = {}

    # ============================================================
    # Run all profiles
    # ============================================================

    for profile_key, credal in profiles.items():

        print("\n" + "=" * 78)
        print(f"PROFILE: {profile_labels.get(profile_key, profile_key)}")
        print("=" * 78)

        print_full_eadmissibility_diagnostics(
            portfolio_names=portfolio_names,
            state_names=state_names,
            utility_matrix=utility_matrix,
            credal_set=credal,
            tolerance=tol,
        )

        admissible, witnesses = all_e_admissible_actions(
            utility_matrix=utility_matrix,
            credal_set=credal,
            tolerance=tol,
            return_witnesses=True,
        )

        print_eadmissibility_results(
            portfolio_names=portfolio_names,
            admissible_indices=admissible,
            witnesses=witnesses,
            state_names=state_names,
        )

        # (A/B/C)
        profile_to_admissibles[profile_key] = [portfolio_names[i] for i in admissible]


    # ============================================================
    # Summary table
    # ============================================================

    print("\n" + "=" * 78)
    print("SUMMARY (E-admissible by profile)")
    print("=" * 78)

    profile_names = list(profile_to_admissibles.keys())
    col_w = max(14, max(len(p) for p in profile_names) + 2)

    header = "Portfolio".ljust(20) + "".join([p.ljust(col_w) for p in profile_names])
    print(header)
    print("-" * len(header))

    for name in portfolio_names:
        row = name.ljust(20)
        for p in profile_names:
            mark = "YES" if name in profile_to_admissibles[p] else "-"
            row += mark.ljust(col_w)
        print(row)


if __name__ == "__main__":
    main()