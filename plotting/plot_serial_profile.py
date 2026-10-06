# Serial profiling: where the runtime goes (gprof flat profile).
# Simple bar chart of % self-time per function. collide + stream dominate (~95%).

import pandas as pd
import matplotlib.pyplot as plt
import style

style.setup()
df = pd.read_csv(style.data_path("serial_profile.csv"))
df = df.sort_values("percent", ascending=False)

fig, ax = plt.subplots()
bars = ax.bar(df["function"], df["percent"], color="#1f77b4")
# Highlight the two hot functions (the parallelization targets).
for bar, fn in zip(bars, df["function"]):
    if fn in ("collide", "stream"):
        bar.set_color("#d62728")

# Annotate each bar with its value.
for bar, p in zip(bars, df["percent"]):
    ax.text(bar.get_x() + bar.get_width() / 2, p + 0.8,
            f"{p:.1f}%", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Self time (% of total)")
ax.set_title("Serial profile: collide + stream = 95.5% of runtime")
ax.set_ylim(0, 65)
ax.grid(axis="x")
plt.xticks(rotation=20)

style.save(fig, "serial_profile.pdf")
