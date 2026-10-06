# MPI strong scaling for three fixed grid sizes (512/1024/2048^2).
# Produces TWO figures from the same CSV:
#   (1) parallel efficiency E(P) vs P        -> the crossover story
#   (2) speedup S(P) vs P with ideal + Amdahl -> deviation from the ideal
# Speedup and efficiency are COMPUTED here from raw MLUPS, not stored.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import style

style.setup()
df = pd.read_csv(style.data_path("mpi_strong.csv"))
sizes = sorted(df["size"].unique())

# For each size: speedup = MLUPS(P) / MLUPS(1), efficiency = speedup / P.
def per_size(size):
    sub = df[df["size"] == size].sort_values("P")
    base = sub[sub["P"] == 1]["mlups"].iloc[0]
    P = sub["P"].to_numpy()
    speedup = sub["mlups"].to_numpy() / base
    eff = speedup / P
    return P, speedup, eff

# ---------- Figure 1: efficiency ----------
fig1, ax1 = plt.subplots()
for size in sizes:
    P, _, eff = per_size(size)
    ax1.plot(P, eff * 100, "-", marker=style.MARKERS[str(size)],
             color=style.COLORS[str(size)], label=f"{size}$^2$")
ax1.axhline(100, ls="--", color=style.IDEAL_COLOR, label="ideal (100%)")
ax1.set_xscale("log", base=2)
ax1.set_xticks(P)
ax1.set_xticklabels([str(p) for p in P])
ax1.set_xlabel("Number of MPI ranks P")
ax1.set_ylabel("Parallel efficiency (%)")
ax1.set_title("MPI strong scaling: efficiency E(P)")
ax1.legend()
style.save(fig1, "mpi_strong_efficiency.pdf")

# ---------- Figure 2: speedup with ideal + Amdahl ----------
# Amdahl with a small serial fraction s: S(P) = 1 / (s + (1-s)/P).
# The gprof profile of the serial code (serial_profile.csv) puts everything
# outside collide/stream/bounce-back below 1% of the runtime, so we take
# s ~ 0.005, i.e. a parallel fraction alpha = 1 - s = 0.995, and label the
# curve by alpha.
s = 0.005
alpha = 1.0 - s
Pfit = np.array(sorted(df["P"].unique()), dtype=float)
amdahl = 1.0 / (s + (1.0 - s) / Pfit)

fig2, ax2 = plt.subplots()
for size in sizes:
    P, speedup, _ = per_size(size)
    ax2.plot(P, speedup, "-", marker=style.MARKERS[str(size)],
             color=style.COLORS[str(size)], label=f"{size}$^2$")
ax2.plot(Pfit, Pfit, "--", color=style.IDEAL_COLOR, label="ideal S = P")
ax2.plot(Pfit, amdahl, ":", color="black",
         label=f"Amdahl ($\\alpha$ = {alpha})")
ax2.set_xscale("log", base=2)
ax2.set_yscale("log", base=2)
ax2.set_xticks(Pfit)
ax2.set_xticklabels([str(int(p)) for p in Pfit])
ax2.set_xlabel("Number of MPI ranks P")
ax2.set_ylabel("Speedup S(P)")
ax2.set_title("MPI strong scaling: speedup vs ideal / Amdahl")
ax2.legend()
style.save(fig2, "mpi_strong_speedup.pdf")
