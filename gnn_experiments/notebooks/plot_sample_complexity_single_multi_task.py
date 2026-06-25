# %%

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

import matplotlib.ticker as ticker

n_total = 41
x_full = np.arange(n_total)

n_labeled = 20
x = np.array([4, 8, 12, 16, 20, 24])
bfs = np.array([1000, 7000, 7000, 7000, 7000, 7000])
dfs = np.array([1000, 5000, 25000, 50000, 100000, 200000])
mst = np.array([1000, 10000, 20000, 50000, 200000, 400000])


fig, ax = plt.subplots(figsize=(6, 4.5))

#plt.plot(x, sample, lw=5, label=r"$\mathrm{Inference}$" ,linestyle='--', color="black",  markersize=15, marker="D")
plt.plot(x, mst, lw=5, label=r"$\mathrm{Prim~ST}$", color="lightgray", markersize=12, marker="^", linestyle=':')
plt.plot(x, dfs, lw=5, label=r"$\mathrm{Dijkstra~ST}$", color="royalblue", markersize=12, marker="s", linestyle='--')
plt.plot(x, bfs, lw=5, label=r"$\mathrm{BFS~ST}$", color="red", markersize=12, marker="o")

font_size = 26
plt.xticks(x,fontsize=font_size)
plt.yticks([0, 1e5, 2e5, 3e5, 4e5], fontsize=font_size)

ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,3))
ax.yaxis.get_offset_text().set_fontsize(font_size)

#plt.ylim(-5e3, 5.5e4)

legend = ax.legend(
    loc="upper left",
    frameon=True,
    fontsize=22,
    handlelength=2.0,
    borderpad=0.4,
    labelspacing=0.4,
)
legend.get_frame().set_linewidth(1.0)

# plt.title(r'$\mathrm{Linear~regression}$', fontsize=30)
plt.xlabel(r'$\mathrm{Number~of~Nodes}$', fontsize=font_size)
plt.ylabel(r'$\mathrm{Number~of~Samples}$', fontsize=font_size)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/plot_graph.pdf", format="pdf", dpi=1200)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

import matplotlib.ticker as ticker

n_total = 41
x_full = np.arange(n_total)

n_labeled = 20
x = np.array([4, 8, 12, 16, 20, 24])

bfs = np.array([1000, 7000, 7000, 7000, 7000, 7000])
bfs_mtl = np.array([1000, 10000, 10000, 10000, 10000, 10000])

dfs = np.array([1000, 5000, 25000, 50000, 100000, 200000])
dfs_mtl = np.array([1000, 5000, 30000, 70000, 150000, 300000])

mst = np.array([1000, 10000, 20000, 50000, 200000, 400000])
mst_mtl = np.array([1000, 10000, 30000, 100000, 300000, 500000])


fig, ax = plt.subplots(figsize=(6, 4.5))

#plt.plot(x, sample, lw=5, label=r"$\mathrm{Inference}$" ,linestyle='--', color="black",  markersize=15, marker="D")
#plt.plot(x, bfs, lw=5, label=r"$\mathrm{BFS}$", color="red", markersize=12, marker="o")
#plt.plot(x, bfs_mtl, lw=5, label=r"$\mathrm{BFS~MTL}$", color="red", markersize=12, marker="o", linestyle='--')
plt.plot(x, mst_mtl, lw=5, label=r"$\mathrm{Prim~MT~w/~Dijkstra}$", color="lightgray", marker="^", markersize=12)
plt.plot(x, mst, lw=5, label=r"$\mathrm{Prim~ST}$", color="lightgray", marker="^", markersize=12, linestyle='dotted')
plt.plot(x, dfs_mtl, lw=5, label=r"$\mathrm{Dijkstra~MT~w/~Prim}$", color="royalblue", marker="s", markersize=12)
plt.plot(x, dfs, lw=5, label=r"$\mathrm{Dijkstra~ST}$", color="royalblue", marker="s", markersize=12, linestyle='--')


ax.set_ylim(-25_000, 825_000)

font_size = 26
plt.xticks(x,fontsize=font_size)
plt.yticks([0, 1e5, 3e5,  5e5, 8e5], fontsize=font_size)

ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,3))
ax.yaxis.get_offset_text().set_fontsize(font_size)

#plt.ylim(-5e3, 5.5e4)
#plt.yscale('log')


legend = ax.legend(
    loc="upper left",
    frameon=True,
    fontsize=22,
    handlelength=2.0,
    borderpad=0.4,
    labelspacing=0.4,
)
legend.get_frame().set_linewidth(1.0)

# plt.title(r'$\mathrm{Linear~regression}$', fontsize=30)
plt.xlabel(r'$\mathrm{Number~of~Nodes}$', fontsize=font_size)
plt.ylabel(r'$\mathrm{Number~of~Samples}$', fontsize=font_size)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/plot_mtl_graph.pdf", format="pdf", dpi=1200)
plt.show()