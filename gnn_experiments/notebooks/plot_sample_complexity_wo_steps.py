# %% 
from __future__ import annotations
from pathlib import Path
from matplotlib import rc

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "figures" / "plot_sample_complexity_no_steps.pdf"

NODES = [4, 8, 12, 16, 20, 24]

class args:
    output: Path = DEFAULT_OUTPUT

# Sample-complexity values used in figures/plot_mtl_graph.pdf.
# The lower-is-better curves compare multi-task (MT) and single-task (ST)
# training for MST Prim and Dijkstra over graph size.
SERIES = [
    {
        "label": "Prim ST w/o Steps",
        "samples": [3000, 20000, 60000, 100000, 400000, 800000],
        "color": "#d0d0d0",
        "marker": "^",
        "linestyle": "-",
    },
    {
        "label": "Prim ST",
        "samples": [1_000, 6_000, 25_000, 60_000, 200_000, 400_000],
        "color": "#d0d0d0",
        "marker": "^",
        "linestyle": ":",
    },
    {
        "label": "Dijkstra ST w/o Steps",
        "samples": [2000, 15000, 50000, 100000, 250000, 450000],
        "color": "royalblue",
        "marker": "s",
        "linestyle": "-",
    },
    {
        "label": "Dijkstra ST",
        "samples": [1_000, 5_000, 25_000, 50_000, 100_000, 200_000],
        "color": "royalblue",
        "marker": "s",
        "linestyle": "--",
    },
]


mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "savefig.dpi": 1200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


fig, ax = plt.subplots(figsize=(6, 4.5))

for series in SERIES:
    ax.plot(
        NODES,
        series["samples"],
        label=series["label"],
        color=series["color"],
        marker=series["marker"],
        linestyle=series["linestyle"],
        linewidth=5.5,
        markersize=15,
        markeredgewidth=0,
    )

ax.set_xlabel("Number of Nodes", fontsize=26, labelpad=6)
ax.set_ylabel("Number of Samples", fontsize=26, labelpad=7)
ax.set_xticks(NODES)
ax.set_yticks([0, 100_000, 300_000, 500_000, 800_000])
ax.set_xlim(3, 25)
ax.set_ylim(-25_000, 825_000)

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((5, 5))
ax.yaxis.set_major_formatter(formatter)
ax.ticklabel_format(axis="y", style="sci", scilimits=(5, 5))

ax.tick_params(axis="both", labelsize=26, width=1.4, length=5)
ax.grid(True, color="#bdbdbd", alpha=0.45, linewidth=0.8)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

legend = ax.legend(
    loc="upper left",
    frameon=True,
    facecolor="white",
    edgecolor="#cfcfcf",
    fontsize=22,
    handlelength=2.0,
    borderpad=0.4,
    labelspacing=0.4,
)
legend.get_frame().set_linewidth(1.0)

# set the text font size off the y-axis
ax.yaxis.offsetText.set_fontsize(26)

fig.subplots_adjust(left=0.18, bottom=0.22, right=0.985, top=0.88)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, format=args.output.suffix.lstrip(".") or "pdf")
plt.show()

# %%
