import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
import solver_N8_UV_profiles_version2 as solver


# ============================================================
# Geometry helpers for 3-state simplex (barycentric -> 2D)
# ============================================================

def grid_simplex(step: float = 0.01) -> np.ndarray:
    pts = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        p = i * step
        for j in range(n + 1 - i):
            q = j * step
            r = 1.0 - p - q
            if r < -1e-12:
                continue
            if r < 0:
                r = 0.0
            pts.append([p, q, r])
    return np.asarray(pts, dtype=float)


def bary_to_xy(pqr: np.ndarray) -> np.ndarray:
    """
    Vertices of triangle:
      S_a = (0,0)
      S_b = (1,0)
      S_c = (0.5, sqrt(3)/2)
    Barycentric coords (p,q,r) mapped to 2D:
      x = q + 0.5*r
      y = (sqrt(3)/2)*r
    """
    q = pqr[:, 1]
    r = pqr[:, 2]
    x = q + 0.5 * r
    y = (np.sqrt(3) / 2.0) * r
    return np.column_stack([x, y])


def draw_triangle(ax, labels):
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2.0]], dtype=float)

    ax.plot([V[0, 0], V[1, 0]], [V[0, 1], V[1, 1]], lw=3)
    ax.plot([V[1, 0], V[2, 0]], [V[1, 1], V[2, 1]], lw=3)
    ax.plot([V[2, 0], V[0, 0]], [V[2, 1], V[0, 1]], lw=3)

    ax.text(V[0, 0] - 0.06, V[0, 1] - 0.05, labels[0], fontsize=16)
    ax.text(V[1, 0] + 0.02, V[1, 1] - 0.05, labels[1], fontsize=16)
    ax.text(V[2, 0] - 0.02, V[2, 1] + 0.04, labels[2], fontsize=16)

    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, np.sqrt(3) / 2 + 0.12)
    ax.axis("off")


# ============================================================
# Credal feasibility in the chosen 3-state submodel
# ============================================================

def in_credal(pi3, F3, b_low, b_up, pi_cap=None, tol=1e-12) -> bool:
    """
    Check:
      b_low <= F3 @ pi3 <= b_up
      pi3 >= 0
      sum(pi3) == 1 (we will only test points from simplex grid so ok)
      optional pi_cap: pi_j <= pi_cap
    """
    if np.any(pi3 < -tol):
        return False
    if abs(np.sum(pi3) - 1.0) > 1e-9:
        return False

    v = F3 @ pi3
    if np.any(v < b_low - tol):
        return False
    if np.any(v > b_up + tol):
        return False

    if pi_cap is not None:
        if np.any(pi3 > float(pi_cap) + tol):
            return False

    return True


def reduce_to_states(F_full, state_names, chosen_states):
    idx = [state_names.index(s) for s in chosen_states]
    F3 = np.asarray(F_full, dtype=float)[:, idx]
    return F3, idx


# ============================================================
# Compute convex hull polygon (in 2D) of feasible credal points
# ============================================================

def compute_credal_hull_2d(F3, b_low, b_up, pi_cap, step=0.01):
    grid = grid_simplex(step=step)

    feasible = []
    for pi3 in grid:
        if in_credal(pi3, F3, b_low, b_up, pi_cap):
            feasible.append(pi3)

    feasible = np.asarray(feasible, dtype=float)
    if feasible.shape[0] < 3:
        return feasible, None  # empty or too small for hull

    xy = bary_to_xy(feasible)
    hull = ConvexHull(xy)
    hull_xy = xy[hull.vertices]
    hull_pi = feasible[hull.vertices]

    return feasible, (hull_xy, hull_pi)


# ============================================================
# Plot single profile + chosen state triplet
# ============================================================

def plot_profile_polygon(profile_key: str, chosen_states, step=0.01, out_prefix="FIG"):
    # 1) Load data from solver (THIS is the “comes from solver” guarantee)
    portfolio_names, state_names, utility_matrix, F = solver.get_simulation_5_problem_data()
    profiles, profile_labels = solver.get_profiles(F)

    if profile_key not in profiles:
        raise ValueError(f"profile_key='{profile_key}' not found. Available: {list(profiles.keys())}")

    credal = profiles[profile_key]
    label = profile_labels.get(profile_key, profile_key)

    # 2) Reduce F to the chosen 3 states
    for s in chosen_states:
        if s not in state_names:
            raise ValueError(f"State '{s}' not in state_names={state_names}")

    F3, idx = reduce_to_states(credal["F"], state_names, chosen_states)

    b_low = np.asarray(credal["b_low"], dtype=float)
    b_up = np.asarray(credal["b_up"], dtype=float)
    pi_cap = credal.get("pi_cap", None)

    # 3) Data check print (optional but useful)
    print("\n=== DATA CHECK (from solver) ===")
    print("profile_key:", profile_key)
    print("profile_label:", label)
    print("state_names:", state_names)
    print("chosen_states:", chosen_states)
    print("F shape:", np.asarray(credal["F"]).shape, "F3 shape:", F3.shape)
    print("b_low:", b_low)
    print("b_up :", b_up)
    print("pi_cap:", pi_cap)
    print("===============================\n")

    # 4) Compute hull
    feasible, hull = compute_credal_hull_2d(F3, b_low, b_up, pi_cap, step=step)
    if hull is None:
        print("[WARN] Not enough feasible points to form a convex hull. Try smaller step.")
        return

    hull_xy, hull_pi = hull

    # 5) Plot
    fig, ax = plt.subplots(figsize=(9, 8))
    draw_triangle(ax, chosen_states)

    # NEW: paint optimality regions inside credal set
    plot_optimality_regions(
        ax=ax,
        feasible_pi3=feasible,
        U=utility_matrix,  
        idx_states=idx,
        alpha=0.22,
        size=10
    )

    # polygon only (no fill)
    closed = np.vstack([hull_xy, hull_xy[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="darkred", lw=1.2)

    ax.set_title(f"Credal set convex polygon — {label} ({','.join(chosen_states)})", fontsize=15, pad=14)
    fig.tight_layout()

    out_name = f"{out_prefix}_{profile_key}_{'_'.join(chosen_states)}.png"
    fig.savefig(out_name, dpi=220)
    plt.close(fig)

    print(f"[OK] Saved: {out_name}")

    # 6) Optional vertex check (like your logs)
    print("\n=== HULL VERTICES CHECK ===\n")
    for t, pi3 in enumerate(hull_pi):
        print(f"Vertex {t}: pi({chosen_states[0]},{chosen_states[1]},{chosen_states[2]}) = {np.round(pi3, 4)}")
        print(f"  sum(pi) = {float(np.sum(pi3)):.6f}")
        if pi_cap is not None:
            print(f"  pi_cap max(pi) = {float(np.max(pi3)):.6f}  (cap={pi_cap})")
        else:
            print(f"  pi_cap max(pi) = {float(np.max(pi3)):.6f}  (cap=None)")

        v = F3 @ pi3
        for r in range(F3.shape[0]):
            slack_lo = float(v[r] - b_low[r])
            slack_up = float(b_up[r] - v[r])
            print(
                f"  feature r={r}: value={float(v[r]):.6f}  (lo={float(b_low[r])}, up={float(b_up[r])})"
                f"  slack_lo={slack_lo:.6f}  slack_up={slack_up:.6f}"
            )
        print("")

def winner_action_on_slice(pi3, U, idx_states):
    # U: (n_actions, n_states_full)
    # idx_states: indices de los 3 estados elegidos, por ejemplo [2,3,4]
    # pi3: (3,)
    eu = U[:, idx_states] @ pi3  # (n_actions,)
    return int(np.argmax(eu)), eu

def plot_optimality_regions(ax, feasible_pi3, U, idx_states, alpha=0.22, size=10):
    xy = bary_to_xy(feasible_pi3)

    winners = []
    for pi3 in feasible_pi3:
        w, _ = winner_action_on_slice(pi3, U, idx_states)
        winners.append(w)
    winners = np.asarray(winners, dtype=int)

    # Scatter por grupo (un color automático por acto)
    unique = np.unique(winners)
    for a in unique:
        mask = (winners == a)
        ax.scatter(xy[mask, 0], xy[mask, 1], s=size, alpha=alpha, label=f"Act {a}")


def main():
    """
    Modified behaviour: generate ALL profiles in one run (no CLI required).
    Also supports CLI flags if you still want them, but default is “everything”.
    """
    portfolio_names, state_names, utility_matrix, F = solver.get_simulation_5_problem_data()
    profiles, profile_labels = solver.get_profiles(F)

    # Default projection per profile (automatic-ish sensible defaults)
    # You can change these in one place later or replace with an auto-picker.
    statesets_by_profile = {
        "credal_0": [
            ["S2", "S3", "S5"],
            ["S3", "S4", "S5"],
        ],
        "A": [
            ["S3", "S4", "S5"],
        ],
        "B": [
            ["S3", "S4", "S5"],
            ["S2", "S4", "S5"],
            ["S2", "S3", "S4"],
            ["S1", "S2", "S5"],
        ],
        "C": [
            ["S2", "S3", "S4"],
            ["S1", "S2", "S4"],
            ["S1", "S2", "S5"],
        ],
    }

    step = 0.0005
    out_prefix = "FIG"

    for profile_key in profiles.keys():
        chosen_list = statesets_by_profile.get(profile_key, [["S3", "S4", "S5"]])

        for chosen_states in chosen_list:
            plot_profile_polygon(
                profile_key=profile_key,
                chosen_states=chosen_states,
                step=step,
                out_prefix=out_prefix
            )


if __name__ == "__main__":
    main()
    
    

