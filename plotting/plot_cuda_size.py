# CUDA problem-size sweep (SoA, block=256). MLUPS vs grid size n (square n x n).
# Throughput rises with size and approaches ~81% of the roofline at 4096^2:
# big grids amortize the fixed per-step overhead (5 kernel launches, probe).

import pandas as pd
import matplotlib.pyplot as plt
import style

style.setup()
df = pd.read_csv(style.data_path("cuda_size.csv")).sort_values("n")

fig, ax = plt.subplots()
ax.plot(df["n"], df["mlups"], "o-", color=style.COLORS["2048"],
        label="measured")
ax.axhline(style.ROOFLINE_MLUPS, ls="--", color=style.ROOFLINE_COLOR,
           label=f"roofline (~{style.ROOFLINE_MLUPS:.0f} MLUPS)")

ax.set_xscale("log", base=2)
ax.set_xticks(df["n"])
ax.set_xticklabels([f"{n}$^2$" for n in df["n"]])
ax.set_xlabel("Grid size")
ax.set_ylabel("Throughput (MLUPS)")
ax.set_title("CUDA problem-size sweep (SoA, block=256)")
ax.set_ylim(0, style.ROOFLINE_MLUPS * 1.1)
ax.legend()

style.save(fig, "cuda_size.pdf")
