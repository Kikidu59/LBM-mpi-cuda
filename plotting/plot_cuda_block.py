# CUDA block-size sweep (SoA, 1024^2). MLUPS vs threads-per-block (log x).
# Memory-bound kernel: broad plateau 128-512, block=32 worst (1 warp only).
# Roofline (~3100 MLUPS) drawn as a reference ceiling.

import pandas as pd
import matplotlib.pyplot as plt
import style

style.setup()
df = pd.read_csv(style.data_path("cuda_block.csv")).sort_values("block")

fig, ax = plt.subplots()
ax.plot(df["block"], df["mlups"], "o-", color=style.COLORS["1024"],
        label="measured")
ax.axhline(style.ROOFLINE_MLUPS, ls="--", color=style.ROOFLINE_COLOR,
           label=f"roofline (~{style.ROOFLINE_MLUPS:.0f} MLUPS)")

ax.set_xscale("log", base=2)
ax.set_xticks(df["block"])
ax.set_xticklabels([str(b) for b in df["block"]])
ax.set_xlabel("Threads per block")
ax.set_ylabel("Throughput (MLUPS)")
ax.set_title("CUDA block-size sweep (1024$^2$, SoA)")
ax.set_ylim(0, style.ROOFLINE_MLUPS * 1.1)
ax.legend()

style.save(fig, "cuda_block.pdf")
