# CUDA memory layout: SoA vs AoS (1024^2, block=256). THE headline result.
# Same flops, same logical bytes; only the access pattern differs -> ~3.6x.
# Roofline (~3100 MLUPS) drawn as a reference ceiling.

import pandas as pd
import matplotlib.pyplot as plt
import style

style.setup()
df = pd.read_csv(style.data_path("cuda_layout.csv"))

fig, ax = plt.subplots()
bars = ax.bar(df["layout"], df["mlups"],
              color=[style.COLORS[l] for l in df["layout"]], width=0.5)

for bar, m in zip(bars, df["mlups"]):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 40,
            f"{m:.0f}", ha="center", va="bottom", fontsize=10)

# Roofline ceiling.
ax.axhline(style.ROOFLINE_MLUPS, ls="--", color=style.ROOFLINE_COLOR,
           label=f"roofline (~{style.ROOFLINE_MLUPS:.0f} MLUPS)")

# Speedup annotation between the two bars.
soa = df[df["layout"] == "SoA"]["mlups"].iloc[0]
aos = df[df["layout"] == "AoS"]["mlups"].iloc[0]
ax.text(0.5, max(soa, aos) * 0.55, f"{soa / aos:.1f}x",
        ha="center", fontsize=13, fontweight="bold")

ax.set_ylabel("Throughput (MLUPS)")
ax.set_title("CUDA: SoA vs AoS — coalescing gives ~3.6x")
ax.set_ylim(0, style.ROOFLINE_MLUPS * 1.1)
ax.grid(axis="x")
ax.legend()

style.save(fig, "cuda_layout.pdf")
