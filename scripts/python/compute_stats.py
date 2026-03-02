# Create a reusable Python script for computing Welch t‑tests, Cohen d, and Hedges g

"""
stats_table.py  —  Compute Welch's t‑test, Cohen's d, and Hedges' g from summary
statistics reported as "mean ±95% CI".

Usage examples
--------------
1) Run with the default hard‑coded table (the one from Warley's paper):

    python stats_table.py

2) Provide your own CSV (must contain Method,Mean,CI[,N]) and specify two
   baseline rows to compare against:

    python stats_table.py --csv my_results.csv --baseline_a Contrastive \\
                          --baseline_b FaCoRNet

3) Show help:

    python stats_table.py -h
"""

import math
import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats


def cohens_d(m1, m2, sd1, sd2, n1, n2):
    s_pooled = math.sqrt(
        ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
    )
    return (m1 - m2) / s_pooled


def hedges_g(d, n1, n2):
    df = n1 + n2 - 2
    J = 1 - 3 / (4 * df - 1)
    return d * J


def welch_t(m1, m2, sd1, sd2, n1, n2):
    t = (m1 - m2) / math.sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2)
    df_num = (sd1 ** 2 / n1 + sd2 ** 2 / n2) ** 2
    df_den = ((sd1 ** 2 / n1) ** 2) / (n1 - 1) + ((sd2 ** 2 / n2) ** 2) / (n2 - 1)
    df = df_num / df_den
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return t, df, p


def load_table(args):
    if args.csv:
        df = pd.read_csv(args.csv)
        required = {"Method", "Mean", "CI"}
        missing = required - set(df.columns)
        if missing:
            sys.exit(f"CSV is missing required columns: {', '.join(missing)}")
        if "N" not in df.columns:
            df["N"] = args.n
    else:
        # Default hard‑coded table from Warley's experiments
        data = {
            "Method": [
                "Contrastive",
                "+ HCL",
                "+ Random Sampler",
                "+ Balanced Sampler",
                "+ Random Sampler + HCL",
                "+ Balanced Sampler + HCL",
                "FaCoRNet",
            ],
            "Mean": [79.50, 78.00, 80.60, 80.50, 81.50, 81.50, 80.40],
            "CI": [0.10, 0.10, 0.30, 0.20, 0.30, 0.30, 0.70],
        }
        df = pd.DataFrame(data)
        df["N"] = 5
    return df


def add_sd(df):
    # Convert CI -> SD (±95 % CI so z = 1.96)
    df["SE"] = df["CI"] / 1.96
    df["SD"] = df["SE"] * (df["N"] ** 0.5)
    return df


def compute_all(df, base_a, base_b):
    n = df["N"].iloc[0]  # assume equal N for simplicity

    baseA = df.loc[df["Method"] == base_a].iloc[0]
    baseB = df.loc[df["Method"] == base_b].iloc[0]

    results = []
    for _, row in df.iterrows():
        if row["Method"] in (base_a, base_b):
            continue

        for label, base in [("A", baseA), ("B", baseB)]:
            t, ddf, p = welch_t(
                row["Mean"], base["Mean"], row["SD"], base["SD"], n, n
            )
            d = cohens_d(row["Mean"], base["Mean"], row["SD"], base["SD"], n, n)
            g = hedges_g(d, n, n)
            results.append(
                {
                    "Comparison": f"{row['Method']} vs "
                    f"{base['Method']}",
                    "ΔMean": row["Mean"] - base["Mean"],
                    "t": t,
                    "df": ddf,
                    "p": p,
                    "Cohen_d": d,
                    "Hedges_g": g,
                }
            )
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Compute Welch t‑test, Cohen d, and Hedges g "
        "from summary statistics."
    )
    parser.add_argument("--csv", help="CSV file with Method,Mean,CI[,N]")
    parser.add_argument(
        "--baseline_a",
        default="Contrastive",
        help="First baseline method name (default: Contrastive)",
    )
    parser.add_argument(
        "--baseline_b",
        default="FaCoRNet",
        help="Second baseline method name (default: FaCoRNet)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=5,
        help="Number of runs per method (used if N column is absent)",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the result table as CSV/TSV/xlsx",
    )
    args = parser.parse_args()

    df = add_sd(load_table(args))

    res = compute_all(df, args.baseline_a, args.baseline_b).round(
        {"ΔMean": 2, "t": 3, "df": 2, "p": 6, "Cohen_d": 3, "Hedges_g": 3}
    )

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() in {".xlsx", ".xls"}:
            res.to_excel(out_path, index=False)
        else:
            res.to_csv(out_path, index=False, sep="\t" if out_path.suffix == ".tsv" else ",")
    else:
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()