"""Create one distribution plot and a summary table for each input feature."""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    data = pd.read_csv("data/test.csv")
    groups = {}
    for column in data.select_dtypes(include="number"):
        match = re.fullmatch(r"(.+)_([1-4])", column)
        groups.setdefault(match.group(1) if match else column, []).append(column)

    out = Path("assets/features")
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, columns in groups.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        for column in columns:
            values = data[column].dropna()
            summary.append({"feature": name, "column": column, "count": len(values),
                            "missing": int(data[column].isna().sum()),
                            "min": values.min(), "median": values.median(),
                            "mean": values.mean(), "max": values.max()})
            ax.hist(values, bins=50, histtype="step", linewidth=1.4,
                    label=f"Period {column[-1]}" if len(columns) > 1 else column)
        ax.set(title=name.replace("_", " ").title(), xlabel="Value", ylabel="Number of rows")
        ax.grid(alpha=0.2)
        if len(columns) > 1:
            ax.legend()
        fig.savefig(out / f"{name.replace('_', '-')}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(summary).to_csv("assets/feature-summary.csv", index=False)
    print(f"Created {len(groups)} feature plots from {len(data):,} rows.")


if __name__ == "__main__":
    main()
