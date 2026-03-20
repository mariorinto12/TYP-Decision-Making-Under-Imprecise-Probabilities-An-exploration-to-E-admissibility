import random
import statistics
from typing import Dict, List
import numpy as np
from sklearn.cluster import KMeans


class MonteCarloPortfolio:
    def __init__(self, initial_investment: float, years: int, mean_return: float, volatility: float, simulations: int = 1000):
        self.initial_investment = initial_investment
        self.years = years
        self.mean_return = mean_return
        self.volatility = volatility
        self.simulations = simulations
        self.results: List[float] = []

    def run_simulation(self):
        self.results = []
        for _ in range(self.simulations):
            value = self.initial_investment
            for _ in range(self.years):
                annual_return = random.gauss(self.mean_return, self.volatility)
                value *= (1 + annual_return)
            self.results.append(value)

    def summary(self) -> Dict[str, float]:
        if not self.results:
            return {}
        return {
            "Mean Final Value": round(statistics.mean(self.results), 2),
            "Median Final Value": round(statistics.median(self.results), 2),
            "Best Case": round(max(self.results), 2),
            "Worst Case": round(min(self.results), 2),
            "5th Percentile": round(sorted(self.results)[int(0.05 * self.simulations)], 2),
            "95th Percentile": round(sorted(self.results)[int(0.95 * self.simulations)], 2),
        }
    
    def probability_of_loss(self) -> float:
        if not self.results:
            return 0.0
        losses = [r for r in self.results if r < self.initial_investment]
        return round(len(losses) / len(self.results) * 100, 2)

# Data of annual returns and volatilities per asset
data = {
    "BBVA": {"annual_return": 0.0896, "annual_volatility": 0.3601},
    "ACS": {"annual_return": 0.0999, "annual_volatility": 0.3417},
    "Repsol": {"annual_return": 0.0652, "annual_volatility": 0.3376},
    "Amadeus": {"annual_return": 0.0688, "annual_volatility": 0.3367},
    "Inditex": {"annual_return": 0.0905, "annual_volatility": 0.2818},
    "Telefonica": {"annual_return": -0.0988, "annual_volatility": 0.2809},
    "Iberdrola": {"annual_return": 0.1367, "annual_volatility": 0.2185},
    "Gold": {"annual_return": 0.0849, "annual_volatility": 0.1481},
    "Bonds": {"annual_return": -0.0173, "annual_volatility": 0.0683},
}

# Establishment of eight different portfolios with different risk profiles (weights)
portfolios = {
    "Balanced": { # Equal weights
        "BBVA": 1/9, "ACS": 1/9, "Repsol": 1/9, "Amadeus": 1/9,
        "Inditex": 1/9, "Telefonica": 1/9, "Iberdrola": 1/9,
        "Gold": 1/9, "Bonds": 1/9
    },
    "Conservative": { # Higher weight on low annual volatility assets
        "BBVA": 0.00, "ACS": 0.00, "Repsol": 0.00, "Amadeus": 0.00,
        "Inditex": 0.20, "Telefonica": 0.15, "Iberdrola": 0.25,
        "Gold": 0.25, "Bonds": 0.15
    },
    "Aggressive": { # Equities-focused and no allocation to bonds or gold
        "BBVA": 0.20, "ACS": 0.15, "Repsol": 0.20, "Amadeus": 0.15,
        "Inditex": 0.10, "Telefonica": 0.10, "Iberdrola": 0.00,
        "Gold": 0.00, "Bonds": 0.00
    },
    
    "MarketIndex": { # Weights based on market capitalization (hypothetical)
        "BBVA": 0.18, "ACS": 0.12, "Repsol": 0.14, "Amadeus": 0.10,
        "Inditex": 0.20, "Telefonica": 0.10, "Iberdrola": 0.10,
        "Gold": 0.03, "Bonds": 0.03
    },
    
    "GoldHedge": { # Higher weight on gold and bonds for hedging
        "BBVA": 0.05, "ACS": 0.05, "Repsol": 0.10, "Amadeus": 0.10,
        "Inditex": 0.10, "Telefonica": 0.10, "Iberdrola": 0.10,
        "Gold": 0.30, "Bonds": 0.10
    },
    
    "BondHeavy": { # Very high weight on bonds, minimal on equities
        "BBVA": 0.00, "ACS": 0.00, "Repsol": 0.00, "Amadeus": 0.05,
        "Inditex": 0.05, "Telefonica": 0.10, "Iberdrola": 0.10,
        "Gold": 0.10, "Bonds": 0.60
    },

    "TechTravel": { # Focus on Amadeus and Inditex, which are in tech/travel and consumer sectors
        "BBVA": 0.05, "ACS": 0.05, "Repsol": 0.05, "Amadeus": 0.40,
        "Inditex": 0.30, "Telefonica": 0.10, "Iberdrola": 0.05,
        "Gold": 0.00, "Bonds": 0.00
    },
    
    "AntiCrisis": { # High weight on gold and bonds, minimal on equities, to perform better in crisis scenarios
        "BBVA": 0.00, "ACS": 0.00, "Repsol": 0.00, "Amadeus": 0.05,
        "Inditex": 0.05, "Telefonica": 0.05, "Iberdrola": 0.10,
        "Gold": 0.40, "Bonds": 0.35
    }
} 

# Function to calculate portfolio mean return and volatility
def portfolio_stats(portfolio_name: str, data: dict, portfolios: dict):
    weights = portfolios[portfolio_name]
    # Markowitz Portfolio Mean Return
    portfolio_return = sum(weights[a] * data[a]["annual_return"] for a in weights)
    # Simplified 'Risk' calculation of Markowitz (since covariance ρ=0 because there is no correlation between assets)
    portfolio_volatility = (sum((weights[a] * data[a]["annual_volatility"])**2 for a in weights))**(1/2)
    return portfolio_return, portfolio_volatility

def assign_labels(value, q):
    if value <= q[0]:
        return "low"
    elif value <= q[1]:
        return "mid-low"
    elif value <= q[2]:
        return "mid"
    elif value <= q[3]:
        return "mid-high"
    else:
        return "high"
    
def signed_score(label_value: int) -> int:
    """
    Map ordinal labels 1..5 to a signed score:
    1 -> -2, 2 -> -1, 3 -> 0, 4 -> +1, 5 -> +2
    """
    mapping = {1: -2, 2: -1, 3: 0, 4: 1, 5: 2}
    return mapping[int(label_value)]


def compute_f_definitions(
    centers_numeric: np.ndarray,
    utility_matrix_reduced: Dict[str, List[int]],
    portfolio_names: List[str]
) -> List[Dict[str, float]]:
    """
    Computes the 8 f-definitions per cluster theta_j using:
    - centers_numeric: shape (K, num_portfolios), integer labels 1..5
    - utility_matrix_reduced: dict portfolio -> list of reduced utilities per cluster (length K)
    Returns a list of dicts, one per cluster, with f1..f8.
    """
    num_clusters = centers_numeric.shape[0]
    num_portfolios = centers_numeric.shape[1]

    # Build utility matrix reduced as an array: shape (num_portfolios, K)
    utilities_reduced_array = np.zeros((num_portfolios, num_clusters), dtype=float)
    for i, name in enumerate(portfolio_names):
        utilities_reduced_array[i, :] = np.array(utility_matrix_reduced[name], dtype=float)

    f_per_cluster = []

    for c in range(num_clusters):
        center = centers_numeric[c, :].astype(int)  # labels 1..5 for this cluster

        # === ORDINAL-based f ===
        f1_mean_ordinal = float(np.mean(center))
        f2_prop_favorable = float(np.sum(center >= 4) / num_portfolios)
        f3_score_mean = float(np.mean([signed_score(x) for x in center]))
        f5_median_ordinal = float(np.median(center))
        f6_prop_adverse = float(np.sum(center <= 2) / num_portfolios)
        f7_std_ordinal = float(np.std(center, ddof=0))  # population std

        # === ECONOMIC-based f (using reduced utilities) ===
        utilities_in_cluster = utilities_reduced_array[:, c]
        f4_mean_utility = float(np.mean(utilities_in_cluster))
        f8_range_utility = float(np.max(utilities_in_cluster) - np.min(utilities_in_cluster))

        f_per_cluster.append({
            "cluster_id": c,
            "f1_mean_ordinal": round(f1_mean_ordinal, 4),
            "f2_prop_fav_ge4": round(f2_prop_favorable, 4),
            "f3_score_mean": round(f3_score_mean, 4),
            "f4_mean_utility": round(f4_mean_utility, 4),
            "f5_median_ordinal": round(f5_median_ordinal, 4),
            "f6_prop_bad_le2": round(f6_prop_adverse, 4),
            "f7_std_ordinal": round(f7_std_ordinal, 4),
            "f8_range_utility": round(f8_range_utility, 4),
        })

    return f_per_cluster


def print_f_table(f_per_cluster: List[Dict[str, float]]):
    """
    Pretty prints a table with f1..f8 for each cluster.
    """
    print("\n=== f(θ) DEFINITIONS PER CLUSTER (8 features) ===")
    header = (
        "Cluster".ljust(10) +
        "f1_mean".rjust(10) +
        "f2_ge4".rjust(10) +
        "f3_score".rjust(10) +
        "f4_u_mean".rjust(12) +
        "f5_med".rjust(10) +
        "f6_le2".rjust(10) +
        "f7_std".rjust(10) +
        "f8_u_rng".rjust(12)
    )
    print(header)
    print("-" * len(header))

    for row in f_per_cluster:
        line = (
            f"{row['cluster_id']}".ljust(10) +
            f"{row['f1_mean_ordinal']:10.4f}" +
            f"{row['f2_prop_fav_ge4']:10.4f}" +
            f"{row['f3_score_mean']:10.4f}" +
            f"{row['f4_mean_utility']:12.4f}" +
            f"{row['f5_median_ordinal']:10.4f}" +
            f"{row['f6_prop_bad_le2']:10.4f}" +
            f"{row['f7_std_ordinal']:10.4f}" +
            f"{row['f8_range_utility']:12.4f}"
        )
        print(line)



def demo():
    
    all_results = {}
    all_mc_objects = {}     
    portfolio_quantiles = {}
    portfolio_labels = {}
    label_to_num = {
    "low": 1,
    "mid-low": 2,
    "mid": 3,
    "mid-high": 4,
    "high": 5
    }

    # Running simulations for each portfolio
    for name in portfolios.keys():
        portfolio_return, portfolio_volatility = portfolio_stats(name, data, portfolios)
        
        portfolio = MonteCarloPortfolio(
            initial_investment = 10000,
            years = 20,
            mean_return = portfolio_return,
            volatility = portfolio_volatility,
            simulations = 5000
        )
        
        # Run the simulation and store results
        portfolio.run_simulation()
        all_results[name] = portfolio.results
        all_mc_objects[name] = portfolio
        
    # Calculate quantiles for each portfolio
    for name, results in all_results.items():
        portfolio_quantiles[name] = np.percentile(
            results, [20, 40, 60, 80]
        )
    # Assign labels based on quantiles
    for name, results in all_results.items():
        q = portfolio_quantiles[name]
        labels = []
        for value in results:
            labels.append(assign_labels(value, q))
        portfolio_labels[name] = labels
        
    # Build matrix X (simulations x portfolios)
    portfolio_names = list(all_results.keys())
    num_simulations = len(next(iter(all_results.values())))

    X = []

    for k in range(num_simulations):
        row = []
        for name in portfolio_names:
            label = portfolio_labels[name][k]
            row.append(label_to_num[label])
        X.append(row)

    X = np.array(X)
    
    X_label = []

    for k in range(num_simulations):
        row = []
        for name in portfolio_names:
            row.append(portfolio_labels[name][k])
        X_label.append(row)

    X_label = np.array(X_label)
    
    
    print("\n=== MONTE CARLO SIMULATION (PER PORTFOLIO) ===")

    for name, mc in all_mc_objects.items():
        print(f"\n{name}")
        print("-" * len(name))

        print("Summary:")
        for k, v in mc.summary().items():
            print(f"  {k}: {v}")

        print(f"Probability of Loss: {mc.probability_of_loss()} %")
        print("First 10 simulated values:")
        print([round(x, 2) for x in mc.results[:10]])

    print("\n--- PORTFOLIO QUINTILES ---")
    for name, q in portfolio_quantiles.items():
        print(f"{name:20s}: {q}")
    
    print("\n--- SAMPLE OUTPUT (first 5 simulations) ---")
    for name in portfolios.keys():
        print(f"\n{name}")
        for i in range(5):
            print(
                f"  Value = {all_results[name][i]:.2f} "
                f"-> State = {portfolio_labels[name][i]}"
        )
    
    print("\n--- MATRIX X INFO ---")
    print("Shape of X (simulations x portfolios):", X.shape)

    print("\n--- FIRST 5 ROWS OF X (numeric) ---")
    for i in range(5):
        print(X[i])

    print("\n--- FIRST 5 ROWS OF X (labels) ---")
    for i in range(5):
        print(X_label[i])

    K = 5
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(X)
    
    unique, counts = np.unique(cluster_ids, return_counts=True)
    print("\n--- CLUSTER SIZES ---")
    for u, c in zip(unique, counts):
        print(f"Cluster {u}: {c}")

    centers = np.rint(km.cluster_centers_).astype(int)

    num_to_label = {v: k for k, v in label_to_num.items()}

    print("\n--- CLUSTER PROFILES (center, rounded) ---")
    for c in range(K):
        center = centers[c]
        labels = [num_to_label[int(x)] for x in center]
        print(f"\nCluster {c}")
        print("  numeric:", center)
        print("  labels :", labels)

        
    # ==============================
    # BUILD UTILITY MATRIX
    # ==============================

    print("\n=== UTILITY MATRIX (mean final value) ===")

    utility_matrix = {}

    for name in portfolio_names:
        utility_matrix[name] = []

        for c in range(K):
            # indices of simulations belonging to cluster c
            idx = np.where(cluster_ids == c)[0]

            # final values of this portfolio in cluster c
            values_in_cluster = [all_results[name][i] for i in idx]

            # utility = mean final value
            utility = np.mean(values_in_cluster)

            utility_matrix[name].append(utility)

    # Print matrix nicely
    header = "Portfolio".ljust(20) + "".join([f"S{c+1}".rjust(12) for c in range(K)])
    print(header)
    print("-" * len(header))

    for name, utils in utility_matrix.items():
        row = name.ljust(20)
        for u in utils:
            row += f"{u:12.2f}"
        print(row)

    print("\n=== UTILITY MATRIX (reduced, integer) ===")
    utility_matrix_reduced = {}

    for name, utils in utility_matrix.items():
        utility_matrix_reduced[name] = [
            int((u // 1000)) for u in utils
        ]

    header = "Portfolio".ljust(20) + "".join([f"S{c+1}".rjust(8) for c in range(K)])
    print(header)
    print("-" * len(header))

    for name, utils in utility_matrix_reduced.items():
        row = name.ljust(20)
        for u in utils:
            row += f"{u:8d}"
        print(row)

    # ==============================
    # COMPUTE f(θ) DEFINITIONS (8)
    # ==============================

    # centers is already computed above as rounded integer centers
    # portfolio_names is already defined earlier
    f_per_cluster = compute_f_definitions(
        centers_numeric=centers,
        utility_matrix_reduced=utility_matrix_reduced,
        portfolio_names=portfolio_names
    )

    print_f_table(f_per_cluster)

    # To map clusters to theta_1..theta_5 explicitly:
    print("\n--- CLUSTER -> THETA MAPPING ---")
    for c in range(K):
        print(f"Cluster {c}  ==  θ{c+1}")
    

if __name__ == "__main__":
    demo()