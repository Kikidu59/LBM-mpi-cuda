# MPI weak scaling: fixed work per rank (128 rows/rank), grow P and grid together.
# Weak efficiency is already in the CSV (= T(1)/T(P)). Ideal weak scaling = flat 100%.

import pandas as pd
import matplotlib.pyplot as plt
import style

style.setup()
df = pd.read_csv(style.data_path("mpi_weak.csv")).sort_values("P")

fig, ax = plt.subplots()
ax.plot(df["P"], df["weak_eff"], "o-", color=style.COLORS["1024"],
        label="measured")
ax.axhline(100, ls="--", color=style.IDEAL_COLOR, label="ideal (100%)")

ax.set_xscale("log", base=2)
ax.set_xticks(df["P"])
ax.set_xticklabels([str(p) for p in df["P"]])
ax.set_xlabel("Number of MPI ranks P (128 rows/rank)")
ax.set_ylabel("Weak efficiency (%)")
ax.set_title("MPI weak scaling: 100% -> 39% (1 -> 128 ranks)")
ax.set_ylim(0, 110)
ax.legend()

style.save(fig, "mpi_weak.pdf")
