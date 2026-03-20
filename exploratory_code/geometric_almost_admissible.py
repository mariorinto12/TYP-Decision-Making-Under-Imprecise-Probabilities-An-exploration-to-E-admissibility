import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea, DrawingArea

import solver_N8_UV_profiles_version2 as solver

PORTFOLIO_COLORS = {
    "Balanced": "tab:blue",
    "Conservative": "tab:green",
    "Aggressive": "tab:red",
    "MarketIndex": "tab:purple",
    "GoldHedge": "tab:brown",
    "BondHeavy": "tab:gray",
    "TechTravel": "tab:orange",
    "AntiCrisis": "tab:pink",
}


# ============================================================
# 1) Barycentric (p,q,r) -> 2D for plotting
# ============================================================

def bary_to_xy(pqr: np.ndarray) -> np.ndarray:
    q = pqr[:, 1]
    r = pqr[:, 2]
    x = q + 0.5 * r
    y = (np.sqrt(3) / 2.0) * r
    return np.column_stack([x, y])


def draw_triangle(ax, labels):
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2.0]], dtype=float)

    ax.plot([V[0, 0], V[1, 0]], [V[0, 1], V[1, 1]], lw=3, color="black")
    ax.plot([V[1, 0], V[2, 0]], [V[1, 1], V[2, 1]], lw=3, color="black")
    ax.plot([V[2, 0], V[0, 0]], [V[2, 1], V[0, 1]], lw=3, color="black")

    ax.text(V[0, 0] - 0.06, V[0, 1] - 0.05, labels[0], fontsize=16)
    ax.text(V[1, 0] + 0.02, V[1, 1] - 0.05, labels[1], fontsize=16)
    ax.text(V[2, 0] - 0.02, V[2, 1] + 0.04, labels[2], fontsize=16)

    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, np.sqrt(3) / 2 + 0.12)
    ax.axis("off")


# ============================================================
# 2) Reduce to chosen states
# ============================================================

def reduce_to_states(F_full, state_names, chosen_states):
    idx = [state_names.index(s) for s in chosen_states]
    F3 = np.asarray(F_full, dtype=float)[:, idx]  # (R x 3)
    return F3, idx


# ============================================================
# 3) Inequalities in (p,q) using r = 1 - p - q
#    We represent each as: a*p + b*q <= c
# ============================================================

def build_credal_inequalities_pq(F3, b_low, b_up, pi_cap=None):
    ineq = []

    # simplex slice:
    # p >= 0  -> -p <= 0
    ineq.append((-1.0, 0.0, 0.0))
    # q >= 0  -> -q <= 0
    ineq.append((0.0, -1.0, 0.0))
    # r >= 0 -> p + q <= 1
    ineq.append((1.0, 1.0, 1.0))

    F3 = np.asarray(F3, dtype=float)
    b_low = np.asarray(b_low, dtype=float).reshape(-1)
    b_up = np.asarray(b_up, dtype=float).reshape(-1)

    R = F3.shape[0]
    for rr in range(R):
        f1, f2, f3 = F3[rr, 0], F3[rr, 1], F3[rr, 2]

        # value = f1*p + f2*q + f3*r
        #       = (f1 - f3)*p + (f2 - f3)*q + f3
        a = f1 - f3
        b = f2 - f3
        const = f3

        # upper: a*p + b*q + const <= b_up  -> a*p + b*q <= b_up - const
        ineq.append((a, b, float(b_up[rr] - const)))

        # lower: a*p + b*q + const >= b_low  -> -(a*p + b*q) <= const - b_low
        ineq.append((-a, -b, float(const - b_low[rr])))

    if pi_cap is not None:
        cap = float(pi_cap)
        # p <= cap
        ineq.append((1.0, 0.0, cap))
        # q <= cap
        ineq.append((0.0, 1.0, cap))
        # r <= cap -> 1 - p - q <= cap -> -p - q <= cap - 1
        ineq.append((-1.0, -1.0, cap - 1.0))

    return ineq


def ineq_from_eu_difference(d123):
    """
    d = (d1,d2,d3) = (U_i - U_k) over the 3 chosen states.
    Condition: d1*p + d2*q + d3*r >= 0 with r=1-p-q

      (d1-d3)p + (d2-d3)q + d3 >= 0
      -> -(d1-d3)p - (d2-d3)q <= d3

    Return (a,b,c) for a*p + b*q <= c.
    """
    d1, d2, d3 = float(d123[0]), float(d123[1]), float(d123[2])
    a = -(d1 - d3)
    b = -(d2 - d3)
    c = d3
    return (a, b, c)


# ============================================================
# 4) Half-plane polygon clipping in (p,q): Sutherland–Hodgman
#    Clip polygon by a*p + b*q <= c
# ============================================================

def clip_polygon_halfplane(poly_pq, halfplane, tol=1e-12):
    """
    poly_pq: (N,2) vertices ordered (convex or not; here convex).
    halfplane: (a,b,c) meaning a*p + b*q <= c.
    Returns clipped polygon vertices (M,2) ordered.
    """
    if poly_pq is None or len(poly_pq) == 0:
        return np.zeros((0, 2), dtype=float)

    a, b, c = halfplane

    def inside(pt):
        return (a * pt[0] + b * pt[1] <= c + tol)

    def intersect(p1, p2):
        # intersection of segment p1->p2 with line a*p + b*q = c
        v = p2 - p1
        denom = a * v[0] + b * v[1]
        if abs(denom) < 1e-15:
            return None
        t = (c - a * p1[0] - b * p1[1]) / denom
        return p1 + t * v

    out = []
    n = len(poly_pq)
    for i in range(n):
        cur = poly_pq[i]
        prev = poly_pq[i - 1]
        cur_in = inside(cur)
        prev_in = inside(prev)

        if cur_in:
            if not prev_in:
                inter = intersect(prev, cur)
                if inter is not None:
                    out.append(inter)
            out.append(cur)
        else:
            if prev_in:
                inter = intersect(prev, cur)
                if inter is not None:
                    out.append(inter)

    return np.asarray(out, dtype=float)


def polygon_area(poly):
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ============================================================
# 5) Build exact credal polygon in (p,q) by clipping simplex triangle
# ============================================================

def build_simplex_triangle_pq():
    """
    In (p,q) with r=1-p-q:
      Vertex r=1 -> (p,q)=(0,0)
      Vertex p=1 -> (1,0)
      Vertex q=1 -> (0,1)
    """
    return np.array([
        [0.0, 0.0],  # r=1
        [1.0, 0.0],  # p=1
        [0.0, 1.0],  # q=1
    ], dtype=float)


def build_credal_polygon_pq(F3, b_low, b_up, pi_cap=None):
    poly = build_simplex_triangle_pq()
    ineq = build_credal_inequalities_pq(F3, b_low, b_up, pi_cap=pi_cap)

    for hp in ineq:
        poly = clip_polygon_halfplane(poly, hp)
        if len(poly) == 0:
            break
    return poly


# ============================================================
# 6) Build exact optimality region for action i
# ============================================================

def build_optimal_region_pq(credal_poly_pq, U3, i, tol=1e-12):
    poly = np.asarray(credal_poly_pq, dtype=float)
    n_actions = U3.shape[0]

    for k in range(n_actions):
        if k == i:
            continue
        d = U3[i] - U3[k]
        hp = ineq_from_eu_difference(d)
        poly = clip_polygon_halfplane(poly, hp, tol=tol)
        if len(poly) == 0:
            break
    return poly


def pq_to_pqr(poly_pq):
    if poly_pq is None or len(poly_pq) == 0:
        return np.zeros((0, 3), dtype=float)
    p = poly_pq[:, 0]
    q = poly_pq[:, 1]
    r = 1.0 - p - q
    return np.column_stack([p, q, r])


# ============================================================
# 6.5) NEW — Project a 5D witness to the chosen 3-state triangle
# ============================================================

def project_witness_5d_to_triangle(pi_5d: np.ndarray, idx_3: list, tol=1e-15):
    """
    pi_5d: shape (5,), may sum < 1 (near-miss) or sum == 1 (witness)
    idx_3: indices of the chosen states (len=3)
    Returns pqr (len=3) summing to 1 (conditional on chosen states), or None if no mass in slice.
    """
    if pi_5d is None:
        return None

    pi_5d = np.asarray(pi_5d, dtype=float).reshape(-1)
    if pi_5d.size < max(idx_3) + 1:
        return None

    pi_3 = pi_5d[idx_3].astype(float)
    s = float(pi_3.sum())

    if s <= tol:
        return None

    return pi_3 / s


# ============================================================
# 7) Plotting
# ============================================================

def plot_exact_regions(
    profile_key: str,
    chosen_states,
    out_prefix="FIG_REGIONS_EXACT",
    overlay_near_miss: bool = True,
    near_miss_min_opt: float = 1e-6,
    tol: float = 1e-12
):
    portfolio_names, state_names, U, F = solver.get_simulation_5_problem_data()
    profiles, profile_labels = solver.get_profiles(F)

    credal = profiles[profile_key]
    label = profile_labels.get(profile_key, profile_key)

    F3, idx = reduce_to_states(credal["F"], state_names, chosen_states)
    b_low = np.asarray(credal["b_low"], dtype=float)
    b_up = np.asarray(credal["b_up"], dtype=float)
    pi_cap = credal.get("pi_cap", None)

    U3 = U[:, idx]  # (8,3)

    print("\n=== DATA CHECK ===")
    print("profile:", profile_key, "|", label)
    print("chosen_states:", chosen_states)
    print("idx:", idx)
    print("pi_cap:", pi_cap)
    print("b_low:", b_low)
    print("b_up :", b_up)
    print("F3:\n", F3)
    print("==================\n")

    # 1) Exact credal polygon in (p,q)
    credal_poly_pq = build_credal_polygon_pq(F3, b_low, b_up, pi_cap=pi_cap)

    simplex_area = 0.5
    credal_area = polygon_area(credal_poly_pq)

    if len(credal_poly_pq) < 3 or credal_area <= 1e-14:
        print("[WARN] Credal polygon slice is empty or degenerate.")
        return

    # 2) Exact optimal regions
    regions = []
    for i in range(U3.shape[0]):
        reg_pq = build_optimal_region_pq(credal_poly_pq, U3, i, tol=tol)
        if len(reg_pq) >= 3 and polygon_area(reg_pq) > 1e-12:
            regions.append((i, reg_pq))

    # 2.5) NEW — get near-miss solutions from the 5D solver and project to this triangle
    near_miss_points = []
    if overlay_near_miss:
        U_full = np.asarray(U, dtype=float)
        for i in range(U_full.shape[0]):
            admissible, res = solver.is_e_admissible(
                utility_matrix=U_full,
                action_index=i,
                credal_set=credal,
                tolerance=1e-9
            )

            if admissible:
                continue

            if not res.get("success", False):
                continue

            opt = res.get("optimal_value", None)
            pi_star = res.get("solution", None)

            if (opt is None) or (pi_star is None):
                continue

            if (opt <= near_miss_min_opt) or (opt >= 1.0 - 1e-9):
                continue

            pqr = project_witness_5d_to_triangle(pi_star, idx_3=idx)
            if pqr is None:
                continue

            xy = bary_to_xy(pqr.reshape(1, 3))[0]
            near_miss_points.append((i, xy, float(opt)))

    # 3) Plot
    fig, ax = plt.subplots(figsize=(9, 8))
    draw_triangle(ax, chosen_states)

    # fill each region polygon
    for (i, reg_pq) in regions:
        name = portfolio_names[i]
        color = PORTFOLIO_COLORS.get(name, "lightgray")

        area_i = polygon_area(reg_pq)
        pct_credal = 100.0 * area_i / credal_area if credal_area > 0 else 0.0
        legend_label = f"{i}: {name} ({pct_credal:.2f}% of credal set)"

        reg_pqr = pq_to_pqr(reg_pq)
        reg_xy = bary_to_xy(reg_pqr)

        ax.fill(
            reg_xy[:, 0],
            reg_xy[:, 1],
            color=color,
            alpha=1.0,
            edgecolor="black",
            linewidth=0.5,
            label=legend_label
        )

    # draw credal boundary on top
    credal_pqr = pq_to_pqr(credal_poly_pq)
    credal_xy = bary_to_xy(credal_pqr)
    closed = np.vstack([credal_xy, credal_xy[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="darkred", lw=1.75)

    # NEW — plot near-miss projected points (stars)
    if overlay_near_miss and len(near_miss_points) > 0:
        for (i, xy, opt) in near_miss_points:
            name = portfolio_names[i]
            color = PORTFOLIO_COLORS.get(name, "lightgray")
            ax.scatter(
                [xy[0]], [xy[1]],
                marker="*",
                s=220,
                c=[color],
                edgecolors="black",
                linewidths=0.9,
                zorder=6
            )
            ax.text(
                xy[0] + 0.012,
                xy[1] + 0.008,
                f"{i}",
                fontsize=9,
                weight="bold",
                color="black",
                zorder=7
            )

    # ----------------------------------------------------------
    # Reference areas box (top-left): line icons + text
    # ----------------------------------------------------------
    credal_pct_simplex = 100.0 * credal_area / simplex_area

    def line_icon(color, lw, width=26, height=10):
        da = DrawingArea(width, height, 0, 0)
        da.add_artist(Line2D([0, width], [height / 2, height / 2], color=color, lw=lw))
        return da

    title_box = TextArea("Reference areas", textprops=dict(size=10, weight="bold"))

    row_simplex = HPacker(
        children=[
            line_icon("black", lw=3),
            TextArea("  Simplex boundary: 100% of simplex", textprops=dict(size=9))
        ],
        align="center",
        pad=0,
        sep=4
    )

    row_credal = HPacker(
        children=[
            line_icon("darkred", lw=1.75),
            TextArea(f"  Credal slice boundary: {credal_pct_simplex:.1f}% of simplex", textprops=dict(size=9))
        ],
        align="center",
        pad=0,
        sep=4
    )

    rows = [title_box, row_simplex, row_credal]

    if overlay_near_miss:
        row_nm = HPacker(
            children=[
                TextArea("  ★  Near-miss (projected)", textprops=dict(size=9))
            ],
            align="center",
            pad=0,
            sep=4
        )
        rows.append(row_nm)

    ref_box = VPacker(
        children=rows,
        align="left",
        pad=0,
        sep=3
    )

    anchored = AnchoredOffsetbox(
        loc="upper left",
        child=ref_box,
        pad=0.4,
        frameon=True,
        borderpad=0.5
    )
    ax.add_artist(anchored)

    # title and legend
    if profile_key == "credal_0":
        title_decisor = "Baseline"
    else:
        title_decisor = f"Decisor {profile_key}"

    title_states = ", ".join(chosen_states)

    ax.set_title(
        f"{title_decisor} - {title_states}",
        fontsize=14,
        pad=10
    )

    leg = ax.legend(loc="upper right", fontsize=9, frameon=True, title="Optimality regions")
    leg.get_title().set_fontsize(10)
    fig.tight_layout()

    out_name = f"{out_prefix}_{profile_key}_{'_'.join(chosen_states)}.png"
    fig.savefig(out_name, dpi=1200)
    plt.close(fig)

    print(f"[OK] Saved: {out_name}")
    print("\nActs with non-empty exact regions in this slice:")
    for (i, reg_pq) in regions:
        area = polygon_area(reg_pq)
        pct_simplex = 100.0 * area / simplex_area
        pct_credal = 100.0 * area / credal_area

        print(
            f"  - {i}: {portfolio_names[i]}"
            f"   area={area:.6e}"
            f"   (%simplex={pct_simplex:7.3f}%)"
            f"   (%credal={pct_credal:7.3f}%)"
        )

    print(f"\n[INFO] Credal slice area={credal_area:.6e}  (%simplex={100.0 * credal_area / simplex_area:7.3f}%)")

    if overlay_near_miss and len(near_miss_points) > 0:
        print("\n[INFO] Near-miss projected points in this slice (i, name, opt):")
        for (i, _, opt) in near_miss_points:
            print(f"  - {i}: {portfolio_names[i]}  opt={opt:.6f}")


def main():
    tests_by_profile = {
        "credal_0": [
            ["S1", "S2", "S3"],
            ["S1", "S2", "S4"],
            ["S1", "S2", "S5"],
            ["S1", "S3", "S4"],
            ["S1", "S3", "S5"],
            ["S1", "S4", "S5"],
            ["S2", "S3", "S4"],
            ["S2", "S3", "S5"],
            ["S2", "S4", "S5"],
            ["S3", "S4", "S5"],
        ],
        "A": [
            ["S1", "S2", "S3"],
            ["S1", "S2", "S4"],
            ["S1", "S2", "S5"],
            ["S1", "S3", "S4"],
            ["S1", "S3", "S5"],
            ["S1", "S4", "S5"],
            ["S2", "S3", "S4"],
            ["S2", "S3", "S5"],
            ["S2", "S4", "S5"],
            ["S3", "S4", "S5"],
        ],
        "B": [
            ["S1", "S2", "S3"],
            ["S1", "S2", "S4"],
            ["S1", "S2", "S5"],
            ["S1", "S3", "S4"],
            ["S1", "S3", "S5"],
            ["S1", "S4", "S5"],
            ["S2", "S3", "S4"],
            ["S2", "S3", "S5"],
            ["S2", "S4", "S5"],
            ["S3", "S4", "S5"],
        ],
        "C": [
            ["S1", "S2", "S3"],
            ["S1", "S2", "S4"],
            ["S1", "S2", "S5"],
            ["S1", "S3", "S4"],
            ["S1", "S3", "S5"],
            ["S1", "S4", "S5"],
            ["S2", "S3", "S4"],
            ["S2", "S3", "S5"],
            ["S2", "S4", "S5"],
            ["S3", "S4", "S5"],
        ],
    }

    for profile_key, triplets in tests_by_profile.items():
        for chosen_states in triplets:
            plot_exact_regions(
                profile_key=profile_key,
                chosen_states=chosen_states,
                overlay_near_miss=True
            )


if __name__ == "__main__":
    main()